/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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
#ifndef __prof_err_h__
#define __prof_err_h__

struct et_list;

void initialize_prof_error_table_r(struct et_list **);

void initialize_prof_error_table(void);
#define init_prof_err_tbl initialize_prof_error_table

typedef enum prof_error_number{
	PROF_VERSION = -1429577728,
	PROF_MAGIC_NODE = -1429577727,
	PROF_NO_SECTION = -1429577726,
	PROF_NO_RELATION = -1429577725,
	PROF_ADD_NOT_SECTION = -1429577724,
	PROF_SECTION_WITH_VALUE = -1429577723,
	PROF_BAD_LINK_LIST = -1429577722,
	PROF_BAD_GROUP_LVL = -1429577721,
	PROF_BAD_PARENT_PTR = -1429577720,
	PROF_MAGIC_ITERATOR = -1429577719,
	PROF_SET_SECTION_VALUE = -1429577718,
	PROF_EINVAL = -1429577717,
	PROF_READ_ONLY = -1429577716,
	PROF_SECTION_NOTOP = -1429577715,
	PROF_SECTION_SYNTAX = -1429577714,
	PROF_RELATION_SYNTAX = -1429577713,
	PROF_EXTRA_CBRACE = -1429577712,
	PROF_MISSING_OBRACE = -1429577711,
	PROF_MAGIC_PROFILE = -1429577710,
	PROF_MAGIC_SECTION = -1429577709,
	PROF_TOPSECTION_ITER_NOSUPP = -1429577708,
	PROF_INVALID_SECTION = -1429577707,
	PROF_END_OF_SECTIONS = -1429577706,
	PROF_BAD_NAMESET = -1429577705,
	PROF_NO_PROFILE = -1429577704,
	PROF_MAGIC_FILE = -1429577703,
	PROF_FAIL_OPEN = -1429577702,
	PROF_EXISTS = -1429577701,
	PROF_BAD_BOOLEAN = -1429577700,
	PROF_BAD_INTEGER = -1429577699,
	PROF_MAGIC_FILE_DATA = -1429577698
} prof_error_number;

#define ERROR_TABLE_BASE_prof -1429577728

#define COM_ERR_BINDDOMAIN_prof "heim_com_err-1429577728"

#endif /* __prof_err_h__ */
