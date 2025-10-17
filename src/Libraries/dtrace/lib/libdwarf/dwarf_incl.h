/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#ifndef DWARF_INCL_H
#define DWARF_INCL_H

#include "libdwarfdefs.h"
#include <string.h>

#ifdef HAVE_ELF_H
#include <sys/elf.h>
#endif

#include <limits.h>
#include <libdwarf.h>
#include <dwarf.h>

#include "dwarf_base_types.h"
#include "dwarf_alloc.h"
#include "dwarf_opaque.h"
#include "dwarf_error.h"
#include "dwarf_util.h"
#endif /* DWARF_INCL_H */
