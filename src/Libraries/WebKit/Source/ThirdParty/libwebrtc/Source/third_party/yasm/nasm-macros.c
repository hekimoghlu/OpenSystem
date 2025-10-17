/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
#include <stddef.h>

static const char *nasm_standard_mac[] = {
    "%define __FILE__",
    "%define __LINE__",
    "%define __SECT__ [section .text]",
    "%imacro section 1+.nolist",
    "%define __SECT__ [section %1]",
    "__SECT__",
    "%endmacro",
    "%imacro segment 1+.nolist",
    "%define __SECT__ [segment %1]",
    "__SECT__",
    "%endmacro",
    "%imacro absolute 1+.nolist",
    "%define __SECT__ [absolute %1]",
    "__SECT__",
    "%endmacro",
    "%imacro struc 1-2.nolist 0",
    "%push struc",
    "%define %$strucname %1",
    "[absolute %2]",
    "%$strucname:",
    "%endmacro",
    "%imacro endstruc 0.nolist",
    "%{$strucname}_size EQU $ - %$strucname",
    "%pop",
    "__SECT__",
    "%endmacro",
    "%imacro istruc 1.nolist",
    "%push istruc",
    "%define %$strucname %1",
    "%$strucstart:",
    "%endmacro",
    "%imacro at 1-2+.nolist",
    "times %1-($-%$strucstart) db 0",
    "%2",
    "%endmacro",
    "%imacro iend 0.nolist",
    "times %{$strucname}_size-($-%$strucstart) db 0",
    "%pop",
    "%endmacro",
    "%imacro align 1-2+.nolist nop",
    "%ifidni %2,nop",
    "[align %1]",
    "%else",
    "times ($$-$) & ((%1)-1) %2",
    "%endif",
    "%endmacro",
    "%imacro alignb 1-2+.nolist resb 1",
    "times ($$-$) & ((%1)-1) %2",
    "%endmacro",
    "%imacro extern 1-*.nolist",
    "%rep %0",
    "[extern %1]",
    "%rotate 1",
    "%endrep",
    "%endmacro",
    "%imacro bits 1+.nolist",
    "[bits %1]",
    "%endmacro",
    "%imacro use16 0.nolist",
    "[bits 16]",
    "%endmacro",
    "%imacro use32 0.nolist",
    "[bits 32]",
    "%endmacro",
    "%imacro use64 0.nolist",
    "[bits 64]",
    "%endmacro",
    "%imacro global 1-*.nolist",
    "%rep %0",
    "[global %1]",
    "%rotate 1",
    "%endrep",
    "%endmacro",
    "%imacro common 1-*.nolist",
    "%rep %0",
    "[common %1]",
    "%rotate 1",
    "%endrep",
    "%endmacro",
    "%imacro cpu 1+.nolist",
    "[cpu %1]",
    "%endmacro",
    "%imacro default 1+.nolist",
    "[default %1]",
    "%endmacro",
    "%define __OUTPUT_FORMAT__ __YASM_OBJFMT__",
    NULL
};
