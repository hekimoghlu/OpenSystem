/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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

#define CONST_IDENT_MAX 30
#define IO_IDENT_MAX 30
#define ARGUMENT_MAX 30
#define USER_LABEL_MAX 30

#define EQUIV_INIT_NAME "equiv"

#define write_nv_ident(fp,a) wr_nv_ident_help ((fp), (struct Addrblock *) (a))
#define nv_type(x) nv_type_help ((struct Addrblock *) x)

extern char *c_keywords[];

char*	c_type_decl Argdcl((int, int));
void	declare_new_addr Argdcl((Addrp));
char*	new_arg_length Argdcl((Namep));
char*	new_func_length Argdcl((void));
int	nv_type_help Argdcl((Addrp));
char*	temp_name Argdcl((char*, int, char*));
char*	user_label Argdcl((long int));
