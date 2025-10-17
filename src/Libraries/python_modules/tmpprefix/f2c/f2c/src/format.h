/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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

#define DEF_C_LINE_LENGTH 77
/* actual max will be 79 */

extern int c_output_line_length;	/* max # chars per line in C source
					   code */

chainp	data_value Argdcl((FILEP, long int, int));
int	do_init_data Argdcl((FILEP, FILEP));
void	list_init_data Argdcl((FILEP*, char*, FILEP));
char*	wr_ardecls Argdcl((FILEP, struct Dimblock*, long int));
void	wr_one_init Argdcl((FILEP, char*, chainp*, int));
void	wr_output_values Argdcl((FILEP, Namep, chainp));
