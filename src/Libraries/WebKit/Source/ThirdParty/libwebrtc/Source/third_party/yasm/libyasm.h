/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#ifndef YASM_LIB_H
#define YASM_LIB_H

#ifdef YASM_PYXELATOR
typedef struct __FILE FILE;
typedef struct __va_list va_list;
typedef unsigned long size_t;
typedef unsigned long uintptr_t;
#else
#include <stdio.h>
#include <stdarg.h>
#include <libyasm-stdint.h>
#endif

#include <libyasm/compat-queue.h>

#include <libyasm/coretype.h>
#include <libyasm/valparam.h>

#include <libyasm/linemap.h>

#include <libyasm/errwarn.h>
#include <libyasm/intnum.h>
#include <libyasm/floatnum.h>
#include <libyasm/expr.h>
#include <libyasm/value.h>
#include <libyasm/symrec.h>

#include <libyasm/bytecode.h>
#include <libyasm/section.h>
#include <libyasm/insn.h>

#include <libyasm/arch.h>
#include <libyasm/dbgfmt.h>
#include <libyasm/objfmt.h>
#include <libyasm/listfmt.h>
#include <libyasm/parser.h>
#include <libyasm/preproc.h>

#include <libyasm/file.h>
#include <libyasm/module.h>

#include <libyasm/hamt.h>
#include <libyasm/md5.h>

#endif
