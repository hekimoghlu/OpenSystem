/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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
#include "internal.h"

#define CALL(n) {void Init_##n(void); Init_##n();}

void
rb_call_inits(void)
{
#if USE_TRANSIENT_HEAP
    CALL(TransientHeap);
#endif
    CALL(vm_postponed_job);
    CALL(Method);
    CALL(RandomSeedCore);
    CALL(sym);
    CALL(var_tables);
    CALL(Object);
    CALL(top_self);
    CALL(Encoding);
    CALL(Comparable);
    CALL(Enumerable);
    CALL(String);
    CALL(Exception);
    CALL(eval);
    CALL(safe);
    CALL(jump);
    CALL(Numeric);
    CALL(Bignum);
    CALL(syserr);
    CALL(Array);
    CALL(Hash);
    CALL(Struct);
    CALL(Regexp);
    CALL(pack);
    CALL(transcode);
    CALL(marshal);
    CALL(Range);
    CALL(IO);
    CALL(Dir);
    CALL(Time);
    CALL(Random);
    CALL(signal);
    CALL(load);
    CALL(Proc);
    CALL(Binding);
    CALL(Math);
    CALL(GC);
    CALL(Enumerator);
    CALL(VM);
    CALL(ISeq);
    CALL(Thread);
    CALL(process);
    CALL(Cont);
    CALL(Rational);
    CALL(Complex);
    CALL(version);
    CALL(vm_trace);
    CALL(vm_stack_canary);
    CALL(ast);
}
#undef CALL
