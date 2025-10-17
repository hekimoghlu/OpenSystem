/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
 * Format of a lesskey file:
 *
 *      LESSKEY_MAGIC (4 bytes)
 *       sections...
 *      END_LESSKEY_MAGIC (4 bytes)
 *
 * Each section is:
 *
 *      section_MAGIC (1 byte)
 *      section_length (2 bytes)
 *      key table (section_length bytes)
 */
#define C0_LESSKEY_MAGIC        '\0'
#define C1_LESSKEY_MAGIC        'M'
#define C2_LESSKEY_MAGIC        '+'
#define C3_LESSKEY_MAGIC        'G'

#define CMD_SECTION             'c'
#define EDIT_SECTION            'e'
#define VAR_SECTION             'v'
#define END_SECTION             'x'

#define C0_END_LESSKEY_MAGIC    'E'
#define C1_END_LESSKEY_MAGIC    'n'
#define C2_END_LESSKEY_MAGIC    'd'

/* */
#define KRADIX          64
