/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#ifndef	_sequence_ext_h_
#define	_sequence_ext_h_

#if defined(__cplusplus)
extern "C" {
#endif

int __seq_stat __P((DB_SEQUENCE *, DB_SEQUENCE_STAT **, u_int32_t));
int __seq_stat_print __P((DB_SEQUENCE *, u_int32_t));
const FN * __db_get_seq_flags_fn __P((void));
const FN * __db_get_seq_flags_fn __P((void));

#if defined(__cplusplus)
}
#endif
#endif /* !_sequence_ext_h_ */
