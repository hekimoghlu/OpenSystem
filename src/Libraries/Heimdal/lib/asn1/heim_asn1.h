/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#ifndef __HEIM_ANY_H__
#define __HEIM_ANY_H__ 1

int	encode_heim_any(unsigned char *, size_t, const heim_any *, size_t *);
int	decode_heim_any(const unsigned char *, size_t, heim_any *, size_t *);
void	free_heim_any(heim_any *);
size_t	length_heim_any(const heim_any *);
int	copy_heim_any(const heim_any *, heim_any *);

int	encode_heim_any_set(unsigned char *, size_t,
			    const heim_any_set *, size_t *);
int	decode_heim_any_set(const unsigned char *, size_t,
			    heim_any_set *,size_t *);
void	free_heim_any_set(heim_any_set *);
size_t	length_heim_any_set(const heim_any_set *);
int	copy_heim_any_set(const heim_any_set *, heim_any_set *);
int	heim_any_cmp(const heim_any_set *, const heim_any_set *);

#endif /* __HEIM_ANY_H__ */
