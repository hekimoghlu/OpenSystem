/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#include "AirLogRegisterPressure.h"

#if ENABLE(B3_JIT)

#include "AirArgInlines.h"
#include "AirCode.h"
#include "AirInstInlines.h"
#include "AirRegLiveness.h"

namespace JSC { namespace B3 { namespace Air {

void logRegisterPressure(Code& code)
{
    const unsigned totalColumns = 200;
    const unsigned registerColumns = 100;
    
    RegLiveness liveness(code);

    for (BasicBlock* block : code) {
        RegLiveness::LocalCalc localCalc(liveness, block);

        block->dumpHeader(WTF::dataFile());

        Vector<CString> instDumps;
        for (unsigned instIndex = block->size(); instIndex--;) {
            Inst& inst = block->at(instIndex);
            Inst* prevInst = block->get(instIndex - 1);

            localCalc.execute(instIndex);

            RegisterSet set;
            set.merge(localCalc.live());
            Inst::forEachDefWithExtraClobberedRegs<Reg>(
                prevInst, &inst,
                [&] (Reg reg, Arg::Role, Bank, Width width, PreservedWidth) {
                    ASSERT(width <= Width64 || Options::useWasmSIMD());
                    set.add(reg, width);
                });

            StringPrintStream instOut;
            StringPrintStream lineOut;
            lineOut.print("   ");
            if (set.numberOfSetRegisters()) {
                set.forEach(
                    [&] (Reg reg) {
                        CString text = toCString(" ", reg);
                        if (text.length() + lineOut.length() > totalColumns) {
                            instOut.print(lineOut.toCString(), "\n");
                            lineOut.reset();
                            lineOut.print("       ");
                        }
                        lineOut.print(text);
                    });
                lineOut.print(":");
            }
            if (lineOut.length() > registerColumns) {
                instOut.print(lineOut.toCString(), "\n");
                lineOut.reset();
            }
            while (lineOut.length() < registerColumns)
                lineOut.print(" ");
            lineOut.print(" ");
            lineOut.print(inst);
            instOut.print(lineOut.toCString(), "\n");
            instDumps.append(instOut.toCString());
        }

        for (unsigned i = instDumps.size(); i--;)
            dataLog(instDumps[i]);
        
        block->dumpFooter(WTF::dataFile());
    }
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

