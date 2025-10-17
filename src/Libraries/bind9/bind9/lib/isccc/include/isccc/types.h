/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
/* $Id: types.h,v 1.10 2007/08/28 07:20:43 tbox Exp $ */

#ifndef ISCCC_TYPES_H
#define ISCCC_TYPES_H 1

/*! \file isccc/types.h */

#include <isc/boolean.h>
#include <isc/int.h>
#include <isc/result.h>

/*% isccc_time_t typedef */
typedef isc_uint32_t isccc_time_t;

/*% isccc_sexpr_t typedef */
typedef struct isccc_sexpr isccc_sexpr_t;
/*% isccc_dottedpair_t typedef */
typedef struct isccc_dottedpair isccc_dottedpair_t;
/*% isccc_symtab_t typedef */
typedef struct isccc_symtab isccc_symtab_t;

/*% iscc region structure */
typedef struct isccc_region {
	unsigned char *		rstart;
	unsigned char *		rend;
} isccc_region_t;

#endif /* ISCCC_TYPES_H */
