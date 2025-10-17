/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
 * stdscan.h	header file for stdscan.c
 */

#ifndef NASM_STDSCAN_H
#define NASM_STDSCAN_H

/* Standard scanner */
struct stdscan_state;

void stdscan_set(const struct stdscan_state *);
const struct stdscan_state *stdscan_get(void);
char * pure_func stdscan_tell(void);
void stdscan_reset(char *buffer);
int stdscan(void *pvt, struct tokenval *tv);
void stdscan_pushback(const struct tokenval *tv);
int nasm_token_hash(const char *token, struct tokenval *tv);
void stdscan_cleanup(void);

#endif
