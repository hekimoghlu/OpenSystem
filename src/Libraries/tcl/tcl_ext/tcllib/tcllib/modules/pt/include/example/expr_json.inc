/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
{
    "pt::grammar::peg" : {
        "rules" : {
            "AddOp"     : {
                "is"   : "\/ {t -} {t +}",
                "mode" : "value"
            },
            "Digit"     : {
                "is"   : "\/ {t 0} {t 1} {t 2} {t 3} {t 4} {t 5} {t 6} {t 7} {t 8} {t 9}",
                "mode" : "value"
            },
            "Expression" : {
                "is"   : "\/ {x {t (} {n Expression} {t )}} {x {n Factor} {* {x {n MulOp} {n Factor}}}}",
                "mode" : "value"
            },
            "Factor"    : {
                "is"   : "x {n Term} {* {x {n AddOp} {n Term}}}",
                "mode" : "value"
            },
            "MulOp"     : {
                "is"   : "\/ {t *} {t \/}",
                "mode" : "value"
            },
            "Number"    : {
                "is"   : "x {? {n Sign}} {+ {n Digit}}",
                "mode" : "value"
            },
            "Sign"      : {
                "is"   : "\/ {t -} {t +}",
                "mode" : "value"
            },
            "Term"      : {
                "is"   : "n Number",
                "mode" : "value"
            }
        },
        "start" : "n Expression"
    }
}
}]
