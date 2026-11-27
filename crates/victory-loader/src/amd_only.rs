//! Fewer exceptions from the AMD-only instructions a title uses: how often
//! each kind is emulated ([`crate::crash`]'s handler), and the ones replaced
//! in the title's code so that they raise none.
//!
//! An exception that reaches a vectored handler under Wine and Rosetta costs
//! tens of microseconds, and titles run these instructions in loops:
//!
//! - `clzero` loops (the two forms the handler does whole, one exception
//!   each) are found when the image is mapped and made to jump to the same
//!   loop with plain stores ([`patch_clzero_loops`], `VICTORY_PATCH_CLZERO=0`:
//!   left to the handler).
//! - `monitorx` and `mwaitx`, wait hints the handler steps over, are replaced
//!   at their first exception by a jump over themselves ([`replace_wait`],
//!   `VICTORY_PATCH_WAITS=0`: emulated each time). Found by the exception and
//!   not by their three bytes, which other instructions' operands hold too.
//! - `extrq` and `insertq` (SSE4a, `66`/`f2 0f 78/79`), emulated a whole
//!   exception each by the handler, are replaced at their first exception by
//!   a jump to the same work in plain instructions ([`replace_sse4a`],
//!   `VICTORY_PATCH_SSE4A=0`: emulated each time). Found by the exception,
//!   not by their bytes, which other instructions' operands hold too; the
//!   plain four-byte register form is too short for the jump and stays with
//!   the handler.
//!
//! Diagnostics builds log the counts per frame ([`log_emulated`]).

/// `cmp rm, reg` between 64-bit registers (`REX.W 39 /r`): (`rm`, `reg`).
pub(crate) fn compare_registers(code: &[u8]) -> Option<(u8, u8)> {
    let &[rex @ 0x48..=0x4f, 0x39, modrm] = code else {
        return None;
    };
    (modrm >> 6 == 3).then_some((modrm & 7 | (rex & 1) << 3, (modrm >> 3) & 7 | (rex & 4) << 1))
}

/// A `clzero` loop as the handler knows it (`crash::clzero`): one of
///
/// ```text
/// top: clzero                 top: mov rax, a
///      add rax, 0x40               add a, 0x40
///      cmp rax, b                  clzero
///      jb  top                     cmp a, b
///                                  jb  top
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClzeroLoop {
    /// Where the loop starts and where the instruction after its `jb` is,
    /// as offsets in the code searched.
    pub top: usize,
    pub end: usize,
    /// The instructions ahead of the stores in a pass (`mov rax, a` and
    /// `add a, 0x40`; none in the first form) and after them (`add rax,
    /// 0x40` in the first form; then `cmp`), as the title has them.
    before: [u8; 7],
    before_len: usize,
    after: [u8; 7],
    after_len: usize,
    /// Where the instruction is whose first bytes the jump takes besides the
    /// loop's first instruction: nothing may jump there.
    pub cut: usize,
}

/// The `clzero` loop whose `clzero` is at `code[at]`, if it is one the
/// handler does whole and neither of its registers is `rsp` (the stores
/// that take the loop's place use the stack for a moment).
pub(crate) fn clzero_loop(code: &[u8], at: usize) -> Option<ClzeroLoop> {
    if code.get(at..at + 3)? != [0x0f, 0x01, 0xfc] {
        return None;
    }
    const RSP: u8 = 4;
    if let Some(&[0x48, 0x83, 0xc0, 0x40, c0, c1, c2, 0x72, 0xf4]) = code.get(at + 3..at + 12)
        && let Some((0, limit)) = compare_registers(&[c0, c1, c2])
        && limit != 0
        && limit != RSP
    {
        return Some(ClzeroLoop {
            top: at,
            end: at + 12,
            before: [0; 7],
            before_len: 0,
            after: [0x48, 0x83, 0xc0, 0x40, c0, c1, c2],
            after_len: 7,
            cut: at + 3,
        });
    }
    let before: [u8; 7] = code.get(at.checked_sub(7)?..at)?.try_into().ok()?;
    let [rex @ (0x48 | 0x4c), 0x89, load, add_rex @ (0x48 | 0x49), 0x83, add, 0x40] = before else {
        return None;
    };
    let &[c0, c1, c2, 0x72, 0xf1] = code.get(at + 3..at + 8)? else {
        return None;
    };
    let (a, b) = compare_registers(&[c0, c1, c2])?;
    let found = load & 0xc7 == 0xc0
        && a == (load >> 3) & 7 | (rex & 4) << 1
        && add & 0xf8 == 0xc0
        && a == add & 7 | (add_rex & 1) << 3
        && a != 0
        && b != 0
        && a != RSP
        && b != RSP;
    found.then_some(ClzeroLoop {
        top: at - 7,
        end: at + 8,
        before,
        before_len: 7,
        after: [c0, c1, c2, 0, 0, 0, 0],
        after_len: 3,
        cut: at - 4,
    })
}

/// Every [`clzero_loop`] of `code`, in order.
pub(crate) fn clzero_loops(code: &[u8]) -> Vec<ClzeroLoop> {
    let mut found = Vec::new();
    let mut at = 0;
    while at + 3 <= code.len() {
        if code[at] == 0x0f
            && let Some(l) = clzero_loop(code, at)
        {
            at = l.end;
            found.push(l);
        } else {
            at += 1;
        }
    }
    found
}

/// Whether bytes near `target` read as a short jump to it (`jcc`, `jmp`,
/// `loop`, `jrcxz` with an 8-bit offset). Bytes, not decoded instructions:
/// it says yes too often, never no wrongly for such a jump.
pub(crate) fn short_jump_to(code: &[u8], target: usize) -> bool {
    let from = target.saturating_sub(129);
    let to = (target + 127).min(code.len().saturating_sub(1));
    (from..to).any(|p| {
        matches!(code[p], 0x70..=0x7f | 0xeb | 0xe0..=0xe3)
            && (p as i64 + 2 + i64::from(code[p + 1] as i8)) == target as i64
    })
}

/// `jmp rel32` at `at` to `to`, if it reaches.
fn jump(at: u64, to: u64) -> Option<[u8; 5]> {
    let rel = i32::try_from(to as i64 - (at as i64 + 5)).ok()?;
    let [a, b, c, d] = rel.to_le_bytes();
    Some([0xe9, a, b, c, d])
}

/// The loop `l` (of code mapped at `code_at`) as code at `stub`, `clzero`
/// done by eight plain stores, ending with a jump to the instruction after
/// the loop. It leaves every register, the flags and memory as the loop
/// does, and so as the handler's emulation of it does:
///
/// ```text
/// pass: [mov rax, a; add a, 0x40]      the second form's, as the title has them
///       push rcx
///       mov  rcx, rax
///       and  rcx, -64                  clzero's line: the 64 bytes holding rax
///       mov  qword [rcx + 8k], 0       k = 0..8
///       pop  rcx                       before the compare, whose register it may be
///       [add rax, 0x40]                the first form's
///       cmp  .., b                     the title's own; the flags are its flags
///       jb   pass
///       jmp  the instruction after the loop
/// ```
///
/// (The eight bytes under the stack pointer are written, which a `clzero`
/// leaves alone; an exception writes far more there, and Windows code keeps
/// nothing below the stack pointer.)
pub(crate) fn clzero_stub(l: &ClzeroLoop, code_at: u64, stub: u64) -> Option<Vec<u8>> {
    let mut code = l.before[..l.before_len].to_vec();
    code.extend([0x51, 0x48, 0x89, 0xc1, 0x48, 0x83, 0xe1, 0xc0]);
    for k in 0..8u8 {
        code.extend([0x48, 0xc7, 0x41, k * 8, 0, 0, 0, 0]);
    }
    code.push(0x59);
    code.extend(&l.after[..l.after_len]);
    let back = i8::try_from(-(code.len() as i64 + 2)).ok()?;
    code.extend([0x72, back as u8]);
    let here = stub + code.len() as u64;
    code.extend(jump(here, code_at + l.end as u64)?);
    Some(code)
}

/// Whether `clzero` loops are replaced when the image is mapped: 0 not
/// (`VICTORY_PATCH_CLZERO=0`), 1 unless bytes nearby read as a jump to the
/// one instruction the replacement cuts in two ([`short_jump_to`]), 2 those
/// too (`VICTORY_PATCH_CLZERO=all`).
fn patches_clzero(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("0") => 0,
        Some("all") => 2,
        _ => 1,
    }
}

/// Whether wait instructions are replaced at their first exception; not
/// with `VICTORY_PATCH_WAITS=0`.
fn patches_waits_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

/// Whether `extrq`/`insertq` are replaced at their first exception; not with
/// `VICTORY_PATCH_SSE4A=0`.
fn patches_sse4a_from(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("0")
}

/// SSE4a's `extrq`/`insertq` where the handler emulates it
/// ([`crate::crash::sse4a`]), decoded from the instruction's bytes: one of
///
/// ```text
/// 66 [rex] 0f 78 /0 ib ib   extrq xmm, len, at   the immediate form
/// f2 [rex] 0f 78 /r ib ib   insertq xmm, xmm, len, at
/// 66 [rex] 0f 79 /r         extrq xmm, xmm       the register form
/// f2 [rex] 0f 79 /r         insertq xmm, xmm
/// ```
///
/// `dst` is the register written and `src` the one read: `insertq`'s
/// destination and source as the modrm names them, and `extrq`'s target,
/// which the immediate form takes from the register the modrm names and the
/// register form from its register field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sse4aOp {
    /// `insertq` (`f2`), not `extrq` (`66`).
    pub insert: bool,
    /// The register whose low qword is written.
    pub dst: u8,
    /// The register whose low (or, for the register form of `insertq`,
    /// high) qword holds the field, or that is read.
    pub src: u8,
    /// The immediate form's own field bytes: (length, index).
    pub immediate: Option<(u8, u8)>,
    /// The instruction's length in bytes.
    pub len: usize,
}

/// The `extrq`/`insertq` at `code[0..]`, if it is one of the four forms
/// ([`Sse4aOp`]); the modrm's register field must be 0 in `extrq`'s
/// immediate form, as only that is ever emitted, and both forms are
/// register-to-register. Exactly what [`crate::crash::sse4a`] emulates.
pub(crate) fn decode_sse4a(code: &[u8]) -> Option<Sse4aOp> {
    let &prefix = code.first()?;
    if !matches!(prefix, 0x66 | 0xf2) {
        return None;
    }
    let (rex, at) = if code.get(1).is_some_and(|b| b & 0xf0 == 0x40) {
        (code[1], 2)
    } else {
        (0, 1)
    };
    if *code.get(at)? != 0x0f {
        return None;
    }
    let immediate = matches!(code.get(at + 1)?, 0x78);
    if !immediate && *code.get(at + 1)? != 0x79 {
        return None;
    }
    let modrm = *code.get(at + 2)?;
    if modrm >> 6 != 3 {
        return None;
    }
    let insert = prefix == 0xf2;
    if immediate && !insert && (modrm >> 3) & 7 != 0 {
        return None;
    }
    let dst = (modrm >> 3) & 7 | (rex & 4) << 1;
    let src = modrm & 7 | (rex & 1) << 3;
    let fields = if immediate {
        Some((*code.get(at + 3)?, *code.get(at + 4)?))
    } else {
        None
    };
    let len = if immediate { at + 5 } else { at + 3 };
    // The register written: the immediate form of extrq works on the
    // register the modrm names, the others on its register field.
    let target = if immediate && !insert { src } else { dst };
    Some(Sse4aOp { insert, dst: target, src, immediate: fields, len })
}

/// What the decoded `extrq`/`insertq` does to the XMM registers, given as
/// (low, high) qwords — the handler's emulation ([`crate::crash::sse4a`]),
/// and what the plain-code stub ([`sse4a_stub`]) does.
pub(crate) fn sse4a_apply(op: &Sse4aOp, xmm: &mut [(u64, u64); 16]) {
    let fields = match op.immediate {
        Some(fields) => fields,
        // The register form reads its field from the source's low qword
        // (extrq) or high qword (insertq).
        None => {
            let f = if op.insert {
                xmm[op.src as usize].1
            } else {
                xmm[op.src as usize].0
            };
            (f as u8, (f >> 8) as u8)
        }
    };
    let n = u32::from(fields.0 & 63);
    let at = u32::from(fields.1 & 63);
    let bits = if n == 0 { 64 } else { n };
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let dst = op.dst as usize;
    let src = op.src as usize;
    if !op.insert {
        xmm[dst].0 = (xmm[dst].0 >> at) & mask;
    } else {
        let value = (xmm[src].0 & mask) << at;
        xmm[dst].0 = xmm[dst].0 & !(mask << at) | value;
    }
}

/// `movq r64, xmm` (`66 [REX.W+R] 0f 7e /r`): the low qword of `xmm` into
/// the GPR (`rax` 0, `rcx` 1, `rdx` 2).
fn movq_gpr_from_xmm(out: &mut Vec<u8>, gpr: u8, xmm: u8) {
    let rex = 0x48 | u8::from(xmm >= 8) << 2;
    out.extend([0x66, rex, 0x0f, 0x7e, 0xc0 | (xmm & 7) << 3 | (gpr & 7)]);
}

/// `pinsrq xmm, r64, 0` (`66 [REX.W+R] 0f 3a 22 /r 00`): the GPR's qword
/// into the low half of `xmm`, its high half kept.
fn pinsrq_xmm_from_gpr(out: &mut Vec<u8>, xmm: u8, gpr: u8) {
    let rex = 0x48 | u8::from(xmm >= 8) << 2;
    out.extend([0x66, rex, 0x0f, 0x3a, 0x22, 0xc0 | (xmm & 7) << 3 | (gpr & 7), 0x00]);
}

/// `pextrq r64, xmm, which` (`66 [REX.W+R] 0f 3a 16 /r ib`): one of `xmm`'s
/// qwords into the GPR.
fn pextrq_gpr_from_xmm(out: &mut Vec<u8>, gpr: u8, xmm: u8, which: u8) {
    let rex = 0x48 | u8::from(xmm >= 8) << 2;
    out.extend([0x66, rex, 0x0f, 0x3a, 0x16, 0xc0 | (xmm & 7) << 3 | (gpr & 7), which]);
}

/// The instruction `op` (at `site`, mapped code) as plain code at `stub`,
/// ending with a jump to the instruction after it. It leaves every register
/// but `rax`, `rcx`, `rdx` — saved and restored, with the flags — and all
/// memory as the instruction does (it changes none but its target's low
/// qword), and so does the handler's emulation of it:
///
/// ```text
/// pushfq; push rax, rcx, rdx
/// the immediate form (the field is the instruction's own bytes, a
/// constant here; its mask is folded in):
///   extrq:   movq rax, src; shr rax, at; movabs rcx, mask; and rax, rcx
///            pinsrq dst, rax, 0
///   insertq: movq rax, src; movabs rcx, mask; and rax, rcx; shl rax, at
///            movq rdx, dst; movabs rcx, clear; and rdx, rcx; or rax, rdx
///            pinsrq dst, rax, 0
/// the register form (the field read at run time from the source's low
/// (extrq) or high (insertq) half; its mask made of it):
///   pextrq rax, src, 0/1; movzx ecx, al; movzx edx, ah; and both, 63
///   neg rcx; add rcx, 64; mov rax, -1; shr rax, cl   the mask (a length
///   of 0 shifts by 64, masked to 0, and leaves all ones)
///   mov rcx, rdx (the index)
///   extrq:   movq rdx, dst; shr rdx, cl; and rdx, rax; pinsrq dst, rdx, 0
///   insertq: movq rdx, src; and rdx, rax; shl rdx, cl
///            shl rax, cl; not rax; movq rcx, dst; and rcx, rax; or rcx, rdx
///            pinsrq dst, rcx, 0
/// pop rdx, rcx, rax; popfq; jmp the instruction after the site
/// ```
pub(crate) fn sse4a_stub(op: &Sse4aOp, site: u64, stub: u64) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(96);
    out.extend([0x9c, 0x50, 0x51, 0x52]); // pushfq; push rax, rcx, rdx
    match op.immediate {
        Some((len_imm, at_imm)) => {
            let n = u64::from(len_imm & 63);
            let bits = if n == 0 { 64 } else { n };
            let mask = if bits == 64 { u64::MAX } else { (1 << bits) - 1 };
            let at = u64::from(at_imm & 63);
            movq_gpr_from_xmm(&mut out, 0, op.src); // rax = the source's low half
            if !op.insert {
                out.extend([0x48, 0xc1, 0xe8, at as u8]); // shr rax, the index
                out.extend([0x48, 0xb9]); // movabs rcx, the mask
                out.extend(mask.to_le_bytes());
                out.extend([0x48, 0x21, 0xc8]); // and rax, rcx
                pinsrq_xmm_from_gpr(&mut out, op.dst, 0);
            } else {
                out.extend([0x48, 0xb9]); // movabs rcx, the mask
                out.extend(mask.to_le_bytes());
                out.extend([0x48, 0x21, 0xc8]); // and rax, rcx
                out.extend([0x48, 0xc1, 0xe0, at as u8]); // shl rax, the index
                movq_gpr_from_xmm(&mut out, 2, op.dst); // rdx = the destination's low half
                let clear = !(mask << at);
                out.extend([0x48, 0xb9]); // movabs rcx, what the field clears
                out.extend(clear.to_le_bytes());
                out.extend([0x48, 0x21, 0xca]); // and rdx, rcx
                out.extend([0x48, 0x09, 0xd0]); // or rax, rdx
                pinsrq_xmm_from_gpr(&mut out, op.dst, 0);
            }
        }
        None => {
            pextrq_gpr_from_xmm(&mut out, 0, op.src, u8::from(op.insert)); // rax = the field's qword
            out.extend([
                0x0f, 0xb6, 0xc8, // movzx ecx, al (the length)
                0x0f, 0xb6, 0xd4, // movzx edx, ah (the index)
                0x83, 0xe1, 0x3f, // and ecx, 63
                0x83, 0xe2, 0x3f, // and edx, 63
                0x48, 0xf7, 0xd9, // neg rcx
                0x48, 0x83, 0xc1, 0x40, // add rcx, 64
                0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, // mov rax, -1
                0x48, 0xd3, 0xe8, // shr rax, cl (the mask)
                0x48, 0x89, 0xd1, // mov rcx, rdx (the index)
            ]);
            if !op.insert {
                movq_gpr_from_xmm(&mut out, 2, op.dst); // rdx = the target's low half
                out.extend([
                    0x48, 0xd3, 0xea, // shr rdx, cl
                    0x48, 0x21, 0xc2, // and rdx, rax
                ]);
                pinsrq_xmm_from_gpr(&mut out, op.dst, 2);
            } else {
                movq_gpr_from_xmm(&mut out, 2, op.src); // rdx = the source's low half
                out.extend([
                    0x48, 0x21, 0xd0, // and rdx, rax (the mask)
                    0x48, 0xd3, 0xe2, // shl rdx, cl (the index)
                    0x48, 0xd3, 0xe0, // shl rax, cl (the mask at the index)
                    0x48, 0xf7, 0xd0, // not rax (what the field clears)
                ]);
                movq_gpr_from_xmm(&mut out, 1, op.dst); // rcx = the destination's low half
                out.extend([
                    0x48, 0x21, 0xc1, // and rcx, rax
                    0x48, 0x09, 0xd1, // or rcx, rdx
                ]);
                pinsrq_xmm_from_gpr(&mut out, op.dst, 1);
            }
        }
    }
    out.extend([0x5a, 0x59, 0x58, 0x9d]); // pop rdx, rcx, rax; popfq
    let here = stub + out.len() as u64;
    out.extend(jump(here, site + op.len as u64)?);
    Some(out)
}

/// An `extrq`/`insertq` site replaced at run time, with what the instruction
/// was, so a thread whose exception was raised before the change still
/// emulates it exactly once ([`replaced_sse4a`]).
#[derive(Clone, Copy)]
pub(crate) struct ReplacedSse4a {
    at: usize,
    op: Sse4aOp,
}

static REPLACED_SSE4A: std::sync::Mutex<Vec<ReplacedSse4a>> = std::sync::Mutex::new(Vec::new());

/// Records the site at `at` as replaced by `op`; false if it was recorded
/// already (another thread got there first).
pub(crate) fn note_replaced_sse4a(at: usize, op: Sse4aOp) -> bool {
    let mut all = REPLACED_SSE4A.lock().unwrap_or_else(|e| e.into_inner());
    if all.iter().any(|r| r.at == at) {
        return false;
    }
    all.push(ReplacedSse4a { at, op });
    true
}

/// The `extrq`/`insertq` site at `address` replaced at run time, if any:
/// what the instruction was.
pub(crate) fn replaced_sse4a(address: usize) -> Option<Sse4aOp> {
    REPLACED_SSE4A
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|r| r.at == address)
        .map(|r| r.op)
}

#[cfg(windows)]
pub use on_windows::*;

#[cfg(windows)]
mod on_windows {
    use super::*;
    use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed, Ordering::SeqCst};
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE, VirtualAlloc,
        VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// `IMAGE_SCN_MEM_EXECUTE`.
    const EXECUTE: u32 = 0x2000_0000;

    /// What the handler emulated, by kind (an exception each).
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Kind {
        /// `clzero`, one line.
        Clzero,
        /// A whole `clzero` loop in one exception.
        ClzeroLoop,
        Monitorx,
        Mwaitx,
        Rdpru,
        /// `clwb`, one instruction.
        Clwb,
        /// A whole `clwb` write-back loop in one exception.
        ClwbLoop,
        /// `extrq`, `insertq`.
        Sse4a,
        /// A wait instruction, or an `extrq`/`insertq` site, that was
        /// replaced before its exception was handled (or that the
        /// translator still runs as it was).
        Replaced,
    }
    const KINDS: usize = Kind::Replaced as usize + 1;

    static EMULATED: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
    /// Nanoseconds in the handler for the exceptions counted.
    #[cfg(feature = "diagnostics")]
    static HANDLER_NS: AtomicU64 = AtomicU64::new(0);
    /// `clzero` loops replaced at load, and wait instructions and
    /// `extrq`/`insertq` replaced at their first exception, so far.
    static LOOPS_REPLACED: AtomicU64 = AtomicU64::new(0);
    static WAITS_REPLACED: AtomicU64 = AtomicU64::new(0);
    static SSE4A_REPLACED: AtomicU64 = AtomicU64::new(0);

    /// The handler emulated one of `kind`.
    pub(crate) fn count(kind: Kind) {
        EMULATED[kind as usize].fetch_add(1, Relaxed);
    }

    /// How many of `kind` the handler emulated since the last
    /// [`log_emulated`].
    #[cfg(test)]
    pub(crate) fn counted(kind: Kind) -> u64 {
        EMULATED[kind as usize].load(Relaxed)
    }

    /// Adds the time since it was made to the handler's time when dropped.
    #[cfg(feature = "diagnostics")]
    pub(crate) struct Timed(pub(crate) std::time::Instant);

    #[cfg(feature = "diagnostics")]
    impl Drop for Timed {
        fn drop(&mut self) {
            HANDLER_NS.fetch_add(self.0.elapsed().as_nanos() as u64, Relaxed);
        }
    }

    /// Logs what the handler emulated since the last call, per frame over
    /// `frames` frames: every one of them was an exception. With the frame
    /// statistics (`victory_gpu::also_log_per_frame`).
    #[cfg(feature = "diagnostics")]
    pub fn log_emulated(frames: f64) {
        let n = |kind: Kind| EMULATED[kind as usize].swap(0, Relaxed) as f64 / frames.max(1.0);
        tracing::info!(
            "  AMD-only instructions emulated per frame (an exception each): clzero {:.1} single lines and {:.1} whole loops, \
             monitorx {:.1}, mwaitx {:.1}, rdpru {:.1}, clwb {:.1} single and {:.1} whole loops, extrq/insertq {:.1}, \
             wait and extrq/insertq sites already replaced {:.1}; {:.3} ms in their handler (the exception's own cost not counted); \
             replaced so far: {} clzero loops at load, {} wait instructions and {} extrq/insertq at their first exception",
            n(Kind::Clzero),
            n(Kind::ClzeroLoop),
            n(Kind::Monitorx),
            n(Kind::Mwaitx),
            n(Kind::Rdpru),
            n(Kind::Clwb),
            n(Kind::ClwbLoop),
            n(Kind::Sse4a),
            n(Kind::Replaced),
            HANDLER_NS.swap(0, Relaxed) as f64 / 1e6 / frames.max(1.0),
            LOOPS_REPLACED.load(Relaxed),
            WAITS_REPLACED.load(Relaxed),
            SSE4A_REPLACED.load(Relaxed),
        );
    }

    /// Whether `monitorx` and `mwaitx` are replaced at their first exception
    /// (`VICTORY_PATCH_WAITS=0`: emulated each time, as before).
    pub fn patches_waits() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| patches_waits_from(std::env::var("VICTORY_PATCH_WAITS").ok().as_deref()))
    }

    /// Whether `extrq` and `insertq` are replaced at their first exception
    /// (`VICTORY_PATCH_SSE4A=0`: emulated each time, as before).
    pub fn patches_sse4a() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| patches_sse4a_from(std::env::var("VICTORY_PATCH_SSE4A").ok().as_deref()))
    }

    /// Pools of memory for the plain code that replaces `extrq`/`insertq`:
    /// (where each is, bytes used) each.
    static SSE4A_POOLS: std::sync::Mutex<Vec<(u64, usize)>> = std::sync::Mutex::new(Vec::new());

    /// Replaces the `extrq`/`insertq` at `at` — where one has just raised an
    /// exception in a title's code — by a jump to the same work in plain
    /// instructions ([`sse4a_stub`]); whether this call did.
    ///
    /// Found by the exception, not by its bytes, which other instructions'
    /// operands hold too. The four-byte register form is too short for the
    /// jump and is left to the handler. Before the bytes change, the site
    /// is recorded ([`note_replaced_sse4a`]), so a thread whose exception
    /// was raised by them and is handled after still emulates the
    /// instruction exactly once, from the record
    /// ([`replaced_sse4a`], which `crash::emulate_amd_only` asks first).
    ///
    /// Safe with other threads running the same bytes: the site's first
    /// byte changes in single-byte stores, from the instruction's prefix
    /// through `int3` (`0xcc`) to the jump's `0xe9`, so a thread fetches
    /// either the old instruction (it faults, and is emulated from the
    /// record), the `int3` (the same), or the whole jump — never a mix,
    /// and no thread can be inside the instruction.
    ///
    /// # Safety
    /// `at` is where an illegal-instruction exception was just raised, in
    /// an image the title loaded (or `TEST_CODE`), and its bytes are
    /// readable up to the end of that image.
    pub(crate) unsafe fn replace_sse4a(at: usize) -> bool {
        if !patches_sse4a() {
            return false;
        }
        // Only a title's own code (or, in tests, `TEST_CODE`): never the
        // host's, or memory that merely holds the bytes. The read is bounded
        // by the image's end, which the next page past may not share.
        #[cfg(test)]
        let known = TEST_CODE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|&&(a, b)| (a..b).contains(&at))
            .map(|&(_, b)| b);
        #[cfg(not(test))]
        let known = None;
        let Some(end) = known.or_else(|| crate::threads::title_image_end(at)) else {
            return false;
        };
        // The whole instruction was just fetched, so its bytes are mapped;
        // seven are read at most, and a truncated form decodes to none.
        let n = 7.min(end - at);
        // SAFETY: `at..at + n` lies in the title's image, mapped and readable.
        let read = unsafe { std::slice::from_raw_parts(at as *const u8, n) };
        let mut original = [0u8; 7];
        original[..n].copy_from_slice(read);
        let Some(op) = decode_sse4a(&original) else {
            return false;
        };
        if op.len > n || op.len < 5 {
            // The four-byte register form: too few bytes for the jump.
            return false;
        }
        let _one = PATCHING.lock().unwrap_or_else(|e| e.into_inner());
        // Another thread may have replaced it while we waited for the lock.
        if replaced_sse4a(at).is_some() {
            return false;
        }
        // Re-read under the lock, and record the site before its bytes
        // change, so a pending exception emulates from this exactly once.
        // SAFETY: as above.
        let read = unsafe { std::slice::from_raw_parts(at as *const u8, n) };
        original[..n].copy_from_slice(read);
        let Some(op) = decode_sse4a(&original) else {
            return false;
        };
        if op.len > n || op.len < 5 || !note_replaced_sse4a(at, op) {
            return false;
        }
        // A pool within a jump's reach of the site for the plain code.
        let (stub, body) = {
            let mut pools = SSE4A_POOLS.lock().unwrap_or_else(|e| e.into_inner());
            let mut which = pools.iter().position(|&(base, used)| {
                used + STUB_ROOM <= POOL && jump(at as u64, base + used as u64).is_some()
            });
            if which.is_none() {
                let Some(base) = near(at as u64) else {
                    return false;
                };
                pools.push((base, 0));
                which = Some(pools.len() - 1);
            }
            let pool = &mut pools[which.unwrap_or_else(|| unreachable!("a pool was just chosen"))];
            let stub = pool.0 + pool.1 as u64;
            let Some(body) = sse4a_stub(&op, at as u64, stub) else {
                return false;
            };
            if pool.1 + body.len() > POOL {
                return false;
            }
            pool.1 += body.len().next_multiple_of(16);
            (stub, body)
        };
        let Some(j) = jump(at as u64, stub) else {
            return false;
        };
        let mut old = 0;
        // SAFETY: the pool is ours and has room for the body; the site is
        // the title's code, writable for the moment of the writes.
        unsafe {
            let mut pool_old = 0;
            if VirtualProtect(stub as *const _, body.len(), PAGE_EXECUTE_READWRITE, &mut pool_old) == 0 {
                return false;
            }
            std::ptr::copy_nonoverlapping(body.as_ptr(), stub as *mut u8, body.len());
            VirtualProtect(stub as *const _, body.len(), PAGE_EXECUTE_READ, &mut pool_old);
            if VirtualProtect(at as *const _, op.len, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                return false;
            }
            let first = AtomicU8::from_ptr(at as *mut u8);
            // `int3` first: one byte, whole to every thread, so nothing
            // fetches a mix of old and new bytes.
            if first.compare_exchange(original[0], 0xcc, SeqCst, SeqCst).is_err() {
                return false;
            }
            std::ptr::copy_nonoverlapping(j[1..].as_ptr(), (at + 1) as *mut u8, 4);
            first.compare_exchange(0xcc, 0xe9, SeqCst, SeqCst).ok();
            VirtualProtect(at as *const _, op.len, old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), stub as *const _, body.len());
            FlushInstructionCache(GetCurrentProcess(), at as *const _, op.len);
        }
        SSE4A_REPLACED.fetch_add(1, Relaxed);
        true
    }

    /// Bytes reserved per plain-code stub in a pool.
    const STUB_ROOM: usize = 128;

    /// One thread at a time changes code: a second one would take the
    /// first one's loosened protection for the page's own.
    static PATCHING: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Code tests let [`replace_wait`] change: (start, end) each.
    #[cfg(test)]
    pub(crate) static TEST_CODE: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

    /// The first byte of what replaces a wait instruction: `jmp` over one
    /// byte (`eb 01`), which with the instruction's last byte left behind it
    /// is three bytes that do nothing.
    pub(crate) const JUMP_OVER: [u8; 2] = [0xeb, 0x01];

    /// Replaces the `monitorx` or `mwaitx` (`0f 01 fa`, `0f 01 fb`) at `at`,
    /// where it has just raised an exception in a title's code, by a jump
    /// over itself (`eb 01 fa`, `eb 01 fb`): whether this call did.
    ///
    /// What the handler does for these two is nothing to registers, flags
    /// or memory, and neither does the jump. The difference is in time: no
    /// exception, and for `mwaitx` no yield to another ready thread
    /// (`SwitchToThread`), so a loop waiting with it turns faster.
    ///
    /// Safe with other threads running the same bytes: one byte changes,
    /// in one store, so a thread fetches either the old instruction (it
    /// faults and is emulated, as ever) or the jump, never a mix, and no
    /// thread can be inside the instruction. A thread whose exception was
    /// raised by the old bytes and handled after the change is known by the
    /// new bytes at its instruction pointer (`crash::emulate_amd_only`).
    ///
    /// # Safety
    /// `at` is where an illegal-instruction exception was just raised, and
    /// its three bytes were read as one of the two instructions.
    pub(crate) unsafe fn replace_wait(at: usize) -> bool {
        if !patches_waits() {
            return false;
        }
        #[cfg(test)]
        let known = TEST_CODE.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, b)| (a..b).contains(&at));
        #[cfg(not(test))]
        let known = false;
        // Only a title's own code: never the host's, or memory that merely
        // holds the bytes.
        if !known && !crate::threads::in_title_image(at) {
            return false;
        }
        let _one = PATCHING.lock().unwrap_or_else(|e| e.into_inner());
        let mut old = 0;
        // SAFETY: mapped code (it just ran), writable for the one store and
        // put back; the store is of one byte, whole to every other thread.
        unsafe {
            if VirtualProtect(at as *const _, 1, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                return false;
            }
            let done = AtomicU8::from_ptr(at as *mut u8).compare_exchange(0x0f, JUMP_OVER[0], SeqCst, SeqCst).is_ok();
            VirtualProtect(at as *const _, 1, old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), at as *const _, 3);
            if done {
                WAITS_REPLACED.fetch_add(1, Relaxed);
            }
            done
        }
    }

    /// Makes the `clzero` loops in the executable sections of the image
    /// mapped at `base` jump to plain stores ([`clzero_stub`]), and returns
    /// how many, and how many it left to the handler because bytes nearby
    /// read as a jump into them ([`short_jump_to`]; none are left with
    /// `VICTORY_PATCH_CLZERO=all`). Nothing with `VICTORY_PATCH_CLZERO=0`.
    /// Before any title thread runs.
    ///
    /// The jump takes the loop's first five bytes: its first instruction
    /// and the start of its second. The rest stays, so code that enters the
    /// loop at its compare still finds it, and jumps back to the top; only
    /// a jump straight to that second instruction would land in the middle
    /// of ours, which is what is looked for.
    pub fn patch_clzero_loops(base: u64, info: &victory_pe::ImageInfo) -> (usize, usize) {
        let mode = patches_clzero(std::env::var("VICTORY_PATCH_CLZERO").ok().as_deref());
        if mode == 0 {
            return (0, 0);
        }
        let (mut patched, mut left) = (0, 0);
        let mut pool = None;
        for s in info.sections.iter().filter(|s| s.characteristics & EXECUTE != 0) {
            let start = base + u64::from(s.rva);
            let len = s.virtual_size.min(s.file_size) as usize;
            // SAFETY: the mapped section, before any title thread runs.
            let (p, l) = unsafe { patch_clzero_loops_in(start, len, base, mode == 2, &mut pool) };
            patched += p;
            left += l;
        }
        seal(pool);
        LOOPS_REPLACED.fetch_add(patched as u64, Relaxed);
        (patched, left)
    }

    /// [`patch_clzero_loops`] of the code `[start, start + len)`, its plain
    /// stores in `pool` (where it is and the bytes used; made near `base`
    /// when first needed); `every`: also the loops bytes nearby read as a
    /// jump into. Returns the loops replaced and left.
    ///
    /// # Safety
    /// The range is mapped code that no thread runs meanwhile.
    pub(crate) unsafe fn patch_clzero_loops_in(
        start: u64,
        len: usize,
        base: u64,
        every: bool,
        pool: &mut Option<(u64, usize)>,
    ) -> (usize, usize) {
        let (mut patched, mut left) = (0, 0);
        // SAFETY: mapped, readable code (the caller's).
        let code = unsafe { std::slice::from_raw_parts(start as *const u8, len) };
        for l in clzero_loops(code) {
            if !every && short_jump_to(code, l.cut) {
                left += 1;
                continue;
            }
            if pool.is_none() {
                *pool = near(base).map(|at| (at, 0));
            }
            let Some((at, used)) = pool.as_mut() else {
                tracing::warn!("no memory near the image for clzero loops: left to the handler");
                left += 1;
                continue;
            };
            let stub = *at + *used as u64;
            let site = start + l.top as u64;
            let (Some(body), Some(jump)) = (clzero_stub(&l, start, stub), jump(site, stub)) else {
                left += 1;
                continue;
            };
            if *used + body.len() > POOL {
                left += 1;
                continue;
            }
            let mut old = 0;
            // SAFETY: our pool has room for the body; the site is the
            // title's code, writable for the moment of the write, which no
            // thread runs.
            unsafe {
                std::ptr::copy_nonoverlapping(body.as_ptr(), stub as *mut u8, body.len());
                if VirtualProtect(site as *const _, jump.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                    left += 1;
                    continue;
                }
                std::ptr::copy_nonoverlapping(jump.as_ptr(), site as *mut u8, jump.len());
                VirtualProtect(site as *const _, jump.len(), old, &mut old);
            }
            *used += body.len().next_multiple_of(16);
            patched += 1;
        }
        (patched, left)
    }

    /// The plain stores are written: their memory becomes code.
    pub(crate) fn seal(pool: Option<(u64, usize)>) {
        if let Some((at, _)) = pool {
            let mut old = 0;
            // SAFETY: our pool, written; then our process's whole cache.
            unsafe {
                VirtualProtect(at as *const _, POOL, PAGE_EXECUTE_READ, &mut old);
                FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
            }
        }
    }

    /// Bytes for the loops' plain stores: 64 KiB, some 600 loops.
    const POOL: usize = 64 << 10;

    /// Memory for code within reach of a `jmp rel32` from the image at
    /// `base`: below it, as the masked stores' (`crate::masked::patch`), or
    /// above it where nothing is free below (an image mapped low).
    fn near(base: u64) -> Option<u64> {
        let below = (1..2048u64).filter_map(|step| base.checked_sub(step << 20));
        let above = (1..2048u64).filter_map(|step| base.checked_add(step << 20));
        below.chain(above).find_map(|want| {
            // SAFETY: a plain allocation at a hint; failure returns null.
            let got = unsafe { VirtualAlloc(want as *const _, POOL, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
            (!got.is_null()).then_some(got as u64)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `clzero; add rax, 0x40; cmp rax, r10; jb`.
    const FIRST: [u8; 12] = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4];
    /// `mov rax, rdx; add rdx, 0x40; clzero; cmp rdx, r8; jb`.
    const SECOND: [u8; 15] =
        [0x48, 0x89, 0xd0, 0x48, 0x83, 0xc2, 0x40, 0x0f, 0x01, 0xfc, 0x4c, 0x39, 0xc2, 0x72, 0xf1];

    #[test]
    fn both_loops_are_found_with_their_ends() {
        let mut code = vec![0x90; 4];
        code.extend(FIRST);
        code.extend([0x90; 3]);
        code.extend(SECOND);
        code.push(0xc3);
        let found = clzero_loops(&code);
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].top, found[0].end, found[0].cut), (4, 16, 7));
        assert_eq!((found[1].top, found[1].end, found[1].cut), (19, 34, 22));
        // A lone clzero, and the loops the handler does one line at a time:
        // against rax itself, with another step, with rax as the limit.
        assert!(clzero_loops(&[0x0f, 0x01, 0xfc, 0xc3]).is_empty());
        let mut own = FIRST;
        own[7..10].copy_from_slice(&[0x48, 0x39, 0xc0]);
        assert!(clzero_loops(&own).is_empty());
        let mut step = FIRST;
        step[6] = 0x20;
        assert!(clzero_loops(&step).is_empty());
        let mut limit = SECOND;
        limit[10..13].copy_from_slice(&[0x48, 0x39, 0xc2]);
        assert!(clzero_loops(&limit).is_empty());
        // Against the stack pointer: left to the handler.
        let mut stack = FIRST;
        stack[7..10].copy_from_slice(&[0x48, 0x39, 0xe0]);
        assert!(clzero_loops(&stack).is_empty());
        // Cut short.
        assert!(clzero_loops(&FIRST[..11]).is_empty());
        assert!(clzero_loops(&SECOND[7..]).is_empty());
    }

    #[test]
    fn the_stores_keep_the_titles_own_instructions() {
        let l = clzero_loop(&FIRST, 0).unwrap();
        let stub = clzero_stub(&l, 0x1_0000, 0x9000).unwrap();
        // push rcx; mov rcx, rax; and rcx, -64; eight stores; pop rcx.
        assert_eq!(stub[..8], [0x51, 0x48, 0x89, 0xc1, 0x48, 0x83, 0xe1, 0xc0]);
        assert_eq!(stub[8..16], [0x48, 0xc7, 0x41, 0, 0, 0, 0, 0]);
        assert_eq!(stub[64..72], [0x48, 0xc7, 0x41, 56, 0, 0, 0, 0]);
        assert_eq!(stub[72], 0x59);
        // add rax, 0x40; cmp rax, r10; jb to the start.
        assert_eq!(stub[73..80], FIRST[3..10]);
        assert_eq!(stub[80..82], [0x72, (-82i8) as u8]);
        // jmp to the instruction after the loop.
        assert_eq!(stub[82], 0xe9);
        let rel = i32::from_le_bytes(stub[83..87].try_into().unwrap());
        assert_eq!(0x9000 + 87 + i64::from(rel), 0x1_0000 + 12);
        assert_eq!(stub.len(), 87);
        // The second form: its mov and add first, its compare after.
        let l = clzero_loop(&SECOND, 7).unwrap();
        let stub = clzero_stub(&l, 0x1_0000, 0x9000).unwrap();
        assert_eq!(stub[..7], SECOND[..7]);
        assert_eq!(stub[7], 0x51);
        assert_eq!(stub[80..83], SECOND[10..13]);
        assert_eq!(stub[83..85], [0x72, (-85i8) as u8]);
        let rel = i32::from_le_bytes(stub[86..90].try_into().unwrap());
        assert_eq!(0x9000 + 90 + i64::from(rel), 0x1_0000 + 15);
        // Out of a jump's reach.
        assert!(clzero_stub(&l, 0x1_0000, 0x9000_0000_0000).is_none());
    }

    #[test]
    fn short_jumps_to_an_address_are_seen() {
        let mut code = vec![0x90; 64];
        assert!(!short_jump_to(&code, 30));
        // jne +8 at 20: to 30.
        code[20..22].copy_from_slice(&[0x75, 8]);
        assert!(short_jump_to(&code, 30));
        assert!(!short_jump_to(&code, 31));
        // jmp -12 at 40: to 30.
        code[20] = 0x90;
        code[40..42].copy_from_slice(&[0xeb, (-12i8) as u8]);
        assert!(short_jump_to(&code, 30));
        // The loops' own `jb` goes to their top, not to what the jump cuts.
        let first = clzero_loop(&FIRST, 0).unwrap();
        assert!(!short_jump_to(&FIRST, first.cut));
        let second = clzero_loop(&SECOND, 7).unwrap();
        assert!(!short_jump_to(&SECOND, second.cut));
    }

    #[test]
    fn the_switches_are_on_unless_zero() {
        assert_eq!(patches_clzero(None), 1);
        assert_eq!(patches_clzero(Some("1")), 1);
        assert_eq!(patches_clzero(Some("0")), 0);
        assert_eq!(patches_clzero(Some("all")), 2);
        assert!(patches_waits_from(None));
        assert!(patches_waits_from(Some("1")));
        assert!(!patches_waits_from(Some(" 0 ")));
        assert!(patches_sse4a_from(None));
        assert!(patches_sse4a_from(Some("1")));
        assert!(!patches_sse4a_from(Some(" 0 ")));
    }

    /// `extrq xmm6, len 8, at 20` (the immediate form).
    const EXTRQ_IMM: [u8; 6] = [0x66, 0x0f, 0x78, 0xc6, 0x08, 0x14];
    /// `insertq xmm0, xmm1, len 16, at 16`.
    const INSERTQ_IMM: [u8; 6] = [0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10];
    /// `extrq xmm8, xmm1` (the register form with a REX).
    const EXTRQ_REG: [u8; 5] = [0x66, 0x44, 0x0f, 0x79, 0xc1];
    /// `insertq xmm0, xmm9`.
    const INSERTQ_REG: [u8; 5] = [0xf2, 0x41, 0x0f, 0x79, 0xc1];

    #[test]
    fn sse4a_forms_decode_with_their_registers_and_lengths() {
        let op = decode_sse4a(&EXTRQ_IMM).unwrap();
        assert_eq!(
            (op.insert, op.dst, op.src, op.immediate, op.len),
            (false, 6, 6, Some((8, 20)), 6)
        );
        let op = decode_sse4a(&INSERTQ_IMM).unwrap();
        assert_eq!(
            (op.insert, op.dst, op.src, op.immediate, op.len),
            (true, 0, 1, Some((16, 16)), 6)
        );
        let op = decode_sse4a(&EXTRQ_REG).unwrap();
        assert_eq!((op.insert, op.dst, op.src, op.immediate, op.len), (false, 8, 1, None, 5));
        let op = decode_sse4a(&INSERTQ_REG).unwrap();
        assert_eq!((op.insert, op.dst, op.src, op.immediate, op.len), (true, 0, 9, None, 5));
        // The register form without a REX: decoded (so it can be left to
        // the handler), though too short to patch.
        assert_eq!(decode_sse4a(&[0x66, 0x0f, 0x79, 0xc1]).unwrap().len, 4);
        // `insertq`'s immediate form keeps its register field: the bits land
        // in the register it names.
        let op = decode_sse4a(&[0xf2, 0x0f, 0x78, 0xc9, 0x10, 0x10]).unwrap();
        assert_eq!((op.insert, op.dst, op.src), (true, 1, 1));
        // The forms' own rules: `extrq`'s immediate form has the register
        // field 0, and both are register-to-register.
        assert!(decode_sse4a(&[0x66, 0x0f, 0x78, 0xc9, 0x08, 0x14]).is_none());
        assert!(decode_sse4a(&[0x66, 0x0f, 0x78, 0x06, 0x08, 0x14]).is_none());
        assert!(decode_sse4a(&[0xf3, 0x0f, 0x79, 0xc1]).is_none());
        assert!(decode_sse4a(&[0x66, 0x0f, 0x7a, 0xc1]).is_none());
        // Cut short.
        assert!(decode_sse4a(&EXTRQ_IMM[..5]).is_none());
        assert!(decode_sse4a(&EXTRQ_IMM[..2]).is_none());
    }

    #[test]
    fn sse4a_apply_matches_the_handler() {
        // extrq xmm0, len 8, at 4: only the field, shifted, lands; the high
        // half is kept.
        let mut xmm = [(0u64, 0u64); 16];
        xmm[0] = (0x1122_3344_5566_7788, 0xaaaa);
        sse4a_apply(&decode_sse4a(&[0x66, 0x0f, 0x78, 0xc0, 0x08, 0x04]).unwrap(), &mut xmm);
        assert_eq!(xmm[0], (0x1122_3344_5566_7788 >> 4 & 0xff, 0xaaaa));
        // insertq xmm0, xmm1, len 16, at 16: the source's masked field into
        // the destination's low qword.
        xmm[1] = (0xabcd, 0);
        sse4a_apply(&decode_sse4a(&INSERTQ_IMM).unwrap(), &mut xmm);
        assert_eq!(xmm[0].0, 0xabcd_0000 | 0x78);
        // The immediate forms reach xmm8..xmm15 through a REX.
        let mut xmm = [(0u64, 0u64); 16];
        xmm[8] = (0x1111_2222_3333_4444, 0);
        xmm[9] = (0xabcd, 0x0800);
        sse4a_apply(&decode_sse4a(&[0xf2, 0x45, 0x0f, 0x78, 0xc1, 0x10, 0x10]).unwrap(), &mut xmm);
        assert_eq!(xmm[8].0, 0x1111_2222_abcd_4444);
        sse4a_apply(&decode_sse4a(&[0x66, 0x41, 0x0f, 0x78, 0xc1, 0x08, 0x04]).unwrap(), &mut xmm);
        assert_eq!(xmm[9].0, 0xbc);
        // The register form reads its field from the source's low half
        // (extrq) or high half (insertq).
        let mut xmm = [(0u64, 0u64); 16];
        xmm[0] = (0x1122_3344_5566_7788, 0);
        xmm[1] = (0x0804, 0); // length 4, index 8
        sse4a_apply(&decode_sse4a(&[0x66, 0x0f, 0x79, 0xc1]).unwrap(), &mut xmm); // extrq xmm0, xmm1
        assert_eq!(xmm[0].0, 0x1122_3344_5566_7788 >> 8 & 0xf);
        let mut xmm = [(0u64, 0u64); 16];
        xmm[0] = (0, 0);
        xmm[1] = (0xdead_beef, 0x0804); // high half: length 4, index 8
        sse4a_apply(&decode_sse4a(&[0xf2, 0x0f, 0x79, 0xc1]).unwrap(), &mut xmm); // insertq xmm0, xmm1
        assert_eq!(xmm[0].0, 0xf << 8);
        // A length of 0 is the whole qword.
        let mut xmm = [(0u64, 0u64); 16];
        xmm[0] = (0x1122_3344_5566_7788, 0);
        sse4a_apply(&decode_sse4a(&[0x66, 0x0f, 0x78, 0xc0, 0x00, 0x00]).unwrap(), &mut xmm);
        assert_eq!(xmm[0].0, 0x1122_3344_5566_7788);
    }

    #[test]
    fn the_sse4a_stubs_are_plain_code_that_jumps_back() {
        // pushfq; push rax, rcx, rdx.
        const PROLOGUE: [u8; 4] = [0x9c, 0x50, 0x51, 0x52];
        // pop rdx, rcx, rax; popfq.
        const EPILOGUE: [u8; 4] = [0x5a, 0x59, 0x58, 0x9d];
        for site in [&EXTRQ_IMM[..], &INSERTQ_IMM[..], &EXTRQ_REG[..], &INSERTQ_REG[..]] {
            let op = decode_sse4a(site).unwrap();
            let stub = sse4a_stub(&op, 0x1_0000, 0x9000).unwrap();
            assert_eq!(stub[..4], PROLOGUE, "{site:02x?}");
            let back = stub.len() - 5;
            assert_eq!(stub[back - 4..back], EPILOGUE, "{site:02x?}");
            assert_eq!(stub[back], 0xe9, "{site:02x?}");
            let rel = i32::from_le_bytes(stub[back + 1..].try_into().unwrap());
            assert_eq!(0x9000 + back as i64 + 5 + i64::from(rel), 0x1_0000 + op.len as i64, "{site:02x?}");
            if let Some((len, at)) = op.immediate {
                // movq rax, the source (`66 [rex] 0f 7e ..`).
                assert!(
                    stub.windows(4).any(|w| w == [0x66, 0x48 | u8::from(op.src >= 8) << 2, 0x0f, 0x7e]),
                    "{site:02x?}"
                );
                // The field's mask is folded in as a constant.
                let n = u64::from(len & 63);
                let bits = if n == 0 { 64 } else { n };
                let mask = if bits == 64 { u64::MAX } else { (1 << bits) - 1 };
                let mut movabs = vec![0x48, 0xb9];
                movabs.extend(mask.to_le_bytes());
                assert!(stub.windows(movabs.len()).any(|w| w == movabs), "{site:02x?}");
                if op.insert {
                    let mut clear = vec![0x48, 0xb9];
                    clear.extend((!(mask << u64::from(at & 63))).to_le_bytes());
                    assert!(stub.windows(clear.len()).any(|w| w == clear), "{site:02x?}");
                }
            } else {
                // The register form reads the field with pextrq
                // (`66 [rex] 0f 3a 16 .. ib`).
                assert!(
                    stub.windows(6).any(|w| w[0] == 0x66 && w[2] == 0x0f && w[3] == 0x3a && w[4] == 0x16),
                    "{site:02x?}"
                );
            }
        }
        // The register form without a REX is four bytes: still stubbed only
        // when the site holds the five-byte jump — here the stub is built,
        // and replace_sse4a leaves such sites to the handler.
        assert_eq!(decode_sse4a(&[0x66, 0x0f, 0x79, 0xc1]).unwrap().len, 4);
        // Out of a jump's reach.
        let op = decode_sse4a(&EXTRQ_IMM).unwrap();
        assert!(sse4a_stub(&op, 0x1_0000, 0x9000_0000_0000).is_none());
    }

    #[test]
    fn replaced_sse4a_sites_are_remembered_once() {
        let op = decode_sse4a(&EXTRQ_IMM).unwrap();
        assert!(replaced_sse4a(0x5000).is_none());
        assert!(note_replaced_sse4a(0x5000, op));
        assert!(!note_replaced_sse4a(0x5000, op));
        assert_eq!(replaced_sse4a(0x5000), Some(op));
        assert!(replaced_sse4a(0x5004).is_none());
    }
}
