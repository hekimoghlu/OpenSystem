/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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

#ifndef _CONFIG_H_
#define _CONFIG_H_

/* UTF8 input and output */
#define UTF8_INPUT_ENABLE
#define UTF8_OUTPUT_ENABLE

/* invert characters invalid in Shift_JIS to CP932 */
#define SHIFTJIS_CP932

/* fix input encoding when given by option */
#define INPUT_CODE_FIX

/* --overwrite option */
/* by Satoru Takabayashi <ccsatoru@vega.aichi-u.ac.jp> */
#define OVERWRITE

/* --cap-input, --url-input option */
#define INPUT_OPTION

/* --numchar-input option */
#define NUMCHAR_OPTION

/* --debug, --no-output option */
#define CHECK_OPTION

/* JIS X0212 */
#define X0212_ENABLE

/* --exec-in, --exec-out option
 * require pipe, fork, execvp and so on.
 * please undef this on MS-DOS, MinGW
 * this is still buggy arround child process
 */
/* #define EXEC_IO */

/* Unicode Normalization */
#define UNICODE_NORMALIZATION

/*
 * Select Default Output Encoding
 *
 */

/* #define DEFAULT_CODE_JIS    */
/* #define DEFAULT_CODE_SJIS   */
/* #define DEFAULT_CODE_WINDOWS_31J */
/* #define DEFAULT_CODE_EUC    */
/* #define DEFAULT_CODE_UTF8   */

#endif /* _CONFIG_H_ */
