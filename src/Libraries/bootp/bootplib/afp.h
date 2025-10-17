/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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
 * AFP_PASSWORD_LEN
 * - fixed length, NULL padded to 8 bytes total length
 */
#ifndef _S_AFP_H
#define _S_AFP_H
#define AFP_PASSWORD_LEN		8

/*
 * AFP_PORT_NUMBER
 * - which port the afp server will be listening on
 */
#define AFP_PORT_NUMBER		548

/*
 * AFP_DIRID_NULL
 * - means no directory id
 */
#define AFP_DIRID_NULL		0

/*
 * AFP_DIRID_ROOT
 * - constant value of root directory id
 */
#define AFP_DIRID_ROOT		2

#endif /* _S_AFP_H */
