/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "config.h"
#include "AirTmpWidth.h"

#if ENABLE(B3_JIT)

#include "AirCode.h"
#include "AirInstInlines.h"
#include "AirTmpWidthInlines.h"
#include <wtf/ListDump.h>

namespace JSC { namespace B3 { namespace Air {

TmpWidth::TmpWidth() = default;

TmpWidth::TmpWidth(Code& code)
{
    recompute<GP>(code);
    recompute<FP>(code);
}

TmpWidth::~TmpWidth() = default;

template <Bank bank>
void TmpWidth::recompute(Code& code)
{
    // Set this to true to cause this analysis to always return pessimistic results.
    constexpr bool beCareful = false;
    constexpr bool verbose = false;

    if (verbose) {
        dataLogLn("Code before TmpWidth:");
        dataLog(code);
    }

    auto& bankWidthsVector = widthsVector(bank);
    bankWidthsVector.resize(AbsoluteTmpMapper<bank>::absoluteIndex(code.numTmps(bank)));
    for (unsigned i = 0; i < bankWidthsVector.size(); ++i)
        bankWidthsVector[i] = Widths(bank);
    
    auto assumeTheWorst = [&] (Tmp tmp) {
        if (bank == Arg(tmp).bank()) {
            Width conservative = code.usesSIMD() ? conservativeWidth(bank) : conservativeWidthWithoutVectors(bank);
            addWidths(tmp, { conservative, conservative });
        }
    };
    
    // Assume the worst for registers.
    RegisterSetBuilder::allRegisters().forEach(
        [&] (Reg reg) {
            assumeTheWorst(Tmp(reg));
        });

    if (beCareful) {
        code.forAllTmps(assumeTheWorst);
        
        // We fall through because the fixpoint that follows can only make things even more
        // conservative. This mode isn't meant to be fast, just safe.
    }

    // Now really analyze everything but Move's over Tmp's, but set aside those Move's so we can find
    // them quickly during the fixpoint below. Note that we can make this analysis stronger by
    // recognizing more kinds of Move's or anything that has Move-like behavior, though it's probably not
    // worth it.
    Vector<Inst*> moves;
    for (BasicBlock* block : code) {
        for (Inst& inst : *block) {
            if (inst.kind.opcode == Move && inst.args[1].isTmp()) {
                if (Arg(inst.args[1]).bank() != bank)
                    continue;

                if (inst.args[0].isTmp()) {
                    moves.append(&inst);
                    continue;
                }
                if (inst.args[0].isImm() && inst.args[0].value() >= 0) {
                    Tmp tmp = inst.args[1].tmp();
                    Widths& tmpWidths = widths(tmp);
                    Width maxWidth = Width64;
                    if (inst.args[0].value() <= std::numeric_limits<int8_t>::max())
                        maxWidth = Width8;
                    else if (inst.args[0].value() <= std::numeric_limits<int16_t>::max())
                        maxWidth = Width16;
                    else if (inst.args[0].value() <= std::numeric_limits<int32_t>::max())
                        maxWidth = Width32;

                    tmpWidths.def = std::max(tmpWidths.def, maxWidth);

                    continue;
                }
            }
            inst.forEachTmp(
                [&] (Tmp& tmp, Arg::Role role, Bank tmpBank, Width width) {
                    if (Arg(tmp).bank() != bank)
                        return;

                    Widths& tmpWidths = widths(tmp);
                    if (Arg::isAnyUse(role))
                        tmpWidths.use = std::max(tmpWidths.use, width);

                    if (Arg::isZDef(role))
                        tmpWidths.def = std::max(tmpWidths.def, width);
                    else if (Arg::isAnyDef(role))
                        tmpWidths.def = code.usesSIMD() ? conservativeWidth(tmpBank) : conservativeWidthWithoutVectors(tmpBank);
                });
        }
    }

    // Finally, fixpoint over the Move's.
    bool changed = true;
    while (changed) {
        changed = false;
        for (Inst* move : moves) {
            ASSERT(move->kind.opcode == Move);
            ASSERT(move->args[0].isTmp());
            ASSERT(move->args[1].isTmp());

            Widths& srcWidths = widths(move->args[0].tmp());
            Widths& dstWidths = widths(move->args[1].tmp());

            // Legend:
            //
            //     Move %src, %dst

            // defWidth(%dst) is a promise about how many high bits are zero. The smaller the width, the
            // stronger the promise. This Move may weaken that promise if we know that %src is making a
            // weaker promise. Such forward flow is the only thing that determines defWidth().
            if (dstWidths.def < srcWidths.def) {
                dstWidths.def = srcWidths.def;
                changed = true;
            }

            // srcWidth(%src) is a promise about how many high bits are ignored. The smaller the width,
            // the stronger the promise. This Move may weaken that promise if we know that %dst is making
            // a weaker promise. Such backward flow is the only thing that determines srcWidth().
            if (srcWidths.use < dstWidths.use) {
                srcWidths.use = dstWidths.use;
                changed = true;
            }
        }
    }

    if (verbose) {
        dataLogLn("bank: ", bank, ", widthsVector: ");
        for (unsigned i = 0; i < bankWidthsVector.size(); ++i)
            dataLogLn("\t", AbsoluteTmpMapper<bank>::tmpFromAbsoluteIndex(i), " : ", bankWidthsVector[i]);
    }
}

void TmpWidth::Widths::dump(PrintStream& out) const
{
    out.print("{use = ", use, ", def = ", def, "}");
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

