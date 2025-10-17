/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
/* $Id: xml.h,v 1.4 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_XML_H
#define ISC_XML_H 1

/*
 * This file is here mostly to make it easy to add additional libxml header
 * files as needed across all the users of this file.  Rather than place
 * these libxml includes in each file, one include makes it easy to handle
 * the ifdef as well as adding the ability to add additional functions
 * which may be useful.
 */

#ifdef HAVE_LIBXML2
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>
#endif

#define ISC_XMLCHAR (const xmlChar *)

#define ISC_XML_RENDERCONFIG		0x00000001 /* render config data */
#define ISC_XML_RENDERSTATS		0x00000002 /* render stats */
#define ISC_XML_RENDERALL		0x000000ff /* render everything */

#endif /* ISC_XML_H */
