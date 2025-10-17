/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#include <tar.h>

#include "header_checks.h"

static void tar_h() {
  MACRO(TMAGIC);
  MACRO_VALUE(TMAGLEN, 6);
  MACRO(TVERSION);
  MACRO_VALUE(TVERSLEN, 2);

  MACRO_VALUE(REGTYPE, '0');
  MACRO_VALUE(AREGTYPE, '\0');
  MACRO_VALUE(LNKTYPE, '1');
  MACRO_VALUE(SYMTYPE, '2');
  MACRO_VALUE(CHRTYPE, '3');
  MACRO_VALUE(BLKTYPE, '4');
  MACRO_VALUE(DIRTYPE, '5');
  MACRO_VALUE(FIFOTYPE, '6');
  MACRO_VALUE(CONTTYPE, '7');

  MACRO_VALUE(TSUID, 04000);
  MACRO_VALUE(TSGID, 02000);
  MACRO_VALUE(TSVTX, 01000);
  MACRO_VALUE(TUREAD, 0400);
  MACRO_VALUE(TUWRITE, 0200);
  MACRO_VALUE(TUEXEC, 0100);
  MACRO_VALUE(TGREAD, 040);
  MACRO_VALUE(TGWRITE, 020);
  MACRO_VALUE(TGEXEC, 010);
  MACRO_VALUE(TOREAD, 04);
  MACRO_VALUE(TOWRITE, 02);
  MACRO_VALUE(TOEXEC, 01);
}
