/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
/*
 * Definitions of keys on the PC.
 * Special (non-ASCII) keys on the PC send a two-byte sequence,
 * where the first byte is 0 and the second is as defined below.
 */
#define PCK_SHIFT_TAB           '\017'
#define PCK_ALT_E               '\022'
#define PCK_CAPS_LOCK           '\072'
#define PCK_F1                  '\073'
#define PCK_NUM_LOCK            '\105'
#define PCK_HOME                '\107'
#define PCK_UP                  '\110'
#define PCK_PAGEUP              '\111'
#define PCK_LEFT                '\113'
#define PCK_RIGHT               '\115'
#define PCK_END                 '\117'
#define PCK_DOWN                '\120'
#define PCK_PAGEDOWN            '\121'
#define PCK_INSERT              '\122'
#define PCK_DELETE              '\123'
#define PCK_CTL_LEFT            '\163'
#define PCK_CTL_RIGHT           '\164'
#define PCK_CTL_DELETE          '\223'
