/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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
/* $Id: offset.h,v 1.17 2008/12/01 23:47:45 tbox Exp $ */

#ifndef ISC_OFFSET_H
#define ISC_OFFSET_H 1

/*! \file
 * \brief
 * File offsets are operating-system dependent.
 */
#include <limits.h>             /* Required for CHAR_BIT. */
#include <sys/types.h>
#include <stddef.h>		/* For Linux Standard Base. */

typedef off_t isc_offset_t;

#endif /* ISC_OFFSET_H */
