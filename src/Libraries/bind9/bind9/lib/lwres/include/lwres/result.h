/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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
/* $Id: result.h,v 1.21 2007/06/19 23:47:23 tbox Exp $ */

#ifndef LWRES_RESULT_H
#define LWRES_RESULT_H 1

/*! \file lwres/result.h */

typedef unsigned int lwres_result_t;

#define LWRES_R_SUCCESS			0
#define LWRES_R_NOMEMORY		1
#define LWRES_R_TIMEOUT			2
#define LWRES_R_NOTFOUND		3
#define LWRES_R_UNEXPECTEDEND		4	/* unexpected end of input */
#define LWRES_R_FAILURE			5	/* generic failure */
#define LWRES_R_IOERROR			6
#define LWRES_R_NOTIMPLEMENTED		7
#define LWRES_R_UNEXPECTED		8
#define LWRES_R_TRAILINGDATA		9
#define LWRES_R_INCOMPLETE		10
#define LWRES_R_RETRY			11
#define LWRES_R_TYPENOTFOUND		12
#define LWRES_R_TOOLARGE		13

#endif /* LWRES_RESULT_H */
