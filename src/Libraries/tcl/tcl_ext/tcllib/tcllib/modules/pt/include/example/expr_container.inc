/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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

[example {
snit::type a_pe_grammar {
    constructor {} {
        install myg using pt::peg::container ${selfns}::G
        $myg start {n Expression}
        $myg add   AddOp Digit Expression Factor MulOp Number Sign Term
        $myg modes {
            AddOp      value
            Digit      value
            Expression value
            Factor     value
            MulOp      value
            Number     value
            Sign       value
            Term       value
        }
        $myg rules {
            AddOp      {/ {t -} {t +}}
            Digit      {/ {t 0} {t 1} {t 2} {t 3} {t 4} {t 5} {t 6} {t 7} {t 8} {t 9}}
            Expression {/ {x {t \50} {n Expression} {t \51}} {x {n Factor} {* {x {n MulOp} {n Factor}}}}}
            Factor     {x {n Term} {* {x {n AddOp} {n Term}}}}
            MulOp      {/ {t *} {t /}}
            Number     {x {? {n Sign}} {+ {n Digit}}}
            Sign       {/ {t -} {t +}}
            Term       {n Number}
        }
        return
    }

    component myg
    delegate method * to myg
}
}]
