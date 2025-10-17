/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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

#include <regex.h>

#include "header_checks.h"

static void regex_h() {
  TYPE(regex_t);
  STRUCT_MEMBER(regex_t, size_t, re_nsub);

  TYPE(size_t);

  TYPE(regmatch_t);
  STRUCT_MEMBER(regmatch_t, regoff_t, rm_so);
  STRUCT_MEMBER(regmatch_t, regoff_t, rm_eo);

  MACRO(REG_EXTENDED);
  MACRO(REG_ICASE);
  MACRO(REG_NOSUB);
  MACRO(REG_NEWLINE);

  MACRO(REG_NOTBOL);
  MACRO(REG_NOTEOL);

  MACRO(REG_NOMATCH);
  MACRO(REG_BADPAT);
  MACRO(REG_ECOLLATE);
  MACRO(REG_ECTYPE);
  MACRO(REG_EESCAPE);
  MACRO(REG_ESUBREG);
  MACRO(REG_EBRACK);
  MACRO(REG_EPAREN);
  MACRO(REG_EBRACE);
  MACRO(REG_BADBR);
  MACRO(REG_ERANGE);
  MACRO(REG_ESPACE);
  MACRO(REG_BADRPT);

  FUNCTION(regcomp, int (*f)(regex_t*, const char*, int));
  FUNCTION(regerror, size_t (*f)(int, const regex_t*, char*, size_t));
  FUNCTION(regexec, int (*f)(const regex_t*, const char*, size_t, regmatch_t*, int));
}
