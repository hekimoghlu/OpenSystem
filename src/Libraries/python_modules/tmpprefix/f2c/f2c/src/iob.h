/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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

struct iob_data {
	struct iob_data *next;
	char *type;
	char *name;
	char *fields[1];
	};
struct io_setup {
	char **fields;
	int nelt, type;
	};

struct defines {
	struct defines *next;
	char defname[1];
	};

typedef struct iob_data iob_data;
typedef struct io_setup io_setup;
typedef struct defines defines;

extern iob_data *iob_list;
extern struct Addrblock *io_structs[9];
void	def_start Argdcl((FILEP, char*, char*, char*));
void	new_iob_data Argdcl((io_setup*, char*));
void	other_undefs Argdcl((FILEP));
char*	tostring Argdcl((char*, int));
