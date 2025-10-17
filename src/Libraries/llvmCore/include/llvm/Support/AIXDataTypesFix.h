/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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

//===-- llvm/Support/AIXDataTypesFix.h - Fix datatype defs ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file overrides default system-defined types and limits which cannot be
// done in DataTypes.h.in because it is processed by autoheader first, which
// comments out any #undef statement
//
//===----------------------------------------------------------------------===//

// No include guards desired!

#ifndef SUPPORT_DATATYPES_H
#error "AIXDataTypesFix.h must only be included via DataTypes.h!"
#endif

// GCC is strict about defining large constants: they must have LL modifier.
// These will be defined properly at the end of DataTypes.h
#undef INT64_MAX
#undef INT64_MIN
