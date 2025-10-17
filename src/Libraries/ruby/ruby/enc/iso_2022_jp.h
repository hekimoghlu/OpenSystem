/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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

#include "regenc.h"
/* dummy for unsupported, stateful encoding */
ENC_DUMMY("ISO-2022-JP");
ENC_ALIAS("ISO2022-JP", "ISO-2022-JP");
ENC_REPLICATE("ISO-2022-JP-2", "ISO-2022-JP");
ENC_ALIAS("ISO2022-JP2", "ISO-2022-JP-2");

/*
 * Name: CP50220
 * MIBenum: 2260
 * Link: http://www.iana.org/assignments/charset-reg/CP50220
 *
 * Windows Codepage 50220
 * a ISO-2022-JP variant.
 * This includes
 * * US-ASCII
 * * JIS X 0201 Latin
 * * JIS X 0201 Katakana
 * * JIS X 0208
 * * NEC special characters
 * * NEC selected IBM extended characters
 * and this implementation doesn't include
 * * User Defined Characters
 *
 * So this CP50220 has the same characters of CP51932.
 *
 * See http://legacy-encoding.sourceforge.jp/wiki/index.php?cp50220
 */
ENC_REPLICATE("CP50220", "ISO-2022-JP");

/* Windows Codepage 50221
 * a ISO-2022-JP variant.
 * This includes
 * * US-ASCII
 * * JIS X 0201 Latin
 * * JIS X 0201 Katakana
 * * JIS X 0208
 * * NEC special characters
 * * NEC selected IBM extended characters
 * and this implementation doesn't include
 * * User Defined Characters
 *
 * So this CP50221 has the same characters of CP51932.
 *
 * See http://legacy-encoding.sourceforge.jp/wiki/index.php?cp50221
 */
ENC_REPLICATE("CP50221", "ISO-2022-JP");
