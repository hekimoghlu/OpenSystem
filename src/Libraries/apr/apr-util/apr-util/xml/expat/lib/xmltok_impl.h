/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
enum {
  BT_NONXML,
  BT_MALFORM,
  BT_LT,
  BT_AMP,
  BT_RSQB,
  BT_LEAD2,
  BT_LEAD3,
  BT_LEAD4,
  BT_TRAIL,
  BT_CR,
  BT_LF,
  BT_GT,
  BT_QUOT,
  BT_APOS,
  BT_EQUALS,
  BT_QUEST,
  BT_EXCL,
  BT_SOL,
  BT_SEMI,
  BT_NUM,
  BT_LSQB,
  BT_S,
  BT_NMSTRT,
  BT_COLON,
  BT_HEX,
  BT_DIGIT,
  BT_NAME,
  BT_MINUS,
  BT_OTHER, /* known not to be a name or name start character */
  BT_NONASCII, /* might be a name or name start character */
  BT_PERCNT,
  BT_LPAR,
  BT_RPAR,
  BT_AST,
  BT_PLUS,
  BT_COMMA,
  BT_VERBAR
};

#include <stddef.h>
