/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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
#ifdef __cplusplus
extern "C" {
#endif

/* Registered charset names are at most 40 characters long. */

#define CHARSET_MAX 41

/* Figure out the charset to use from the ContentType.
   buf contains the body of the header field (the part after "Content-Type:").
   charset gets the charset to use.  It must be at least CHARSET_MAX chars
   long.  charset will be empty if the default charset should be used.
*/

void getXMLCharset(const char *buf, char *charset);

#ifdef __cplusplus
}
#endif
