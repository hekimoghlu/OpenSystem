/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
#ifndef __XML_SAVE_H__
#define __XML_SAVE_H__

#include <libxml/tree.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef LIBXML_OUTPUT_ENABLED
void xmlBufAttrSerializeTxtContent(xmlBufPtr buf, xmlDocPtr doc,
                                   xmlAttrPtr attr, const xmlChar * string);
void xmlBufDumpNotationTable(xmlBufPtr buf, xmlNotationTablePtr table);
void xmlBufDumpElementDecl(xmlBufPtr buf, xmlElementPtr elem);
void xmlBufDumpAttributeDecl(xmlBufPtr buf, xmlAttributePtr attr);
void xmlBufDumpEntityDecl(xmlBufPtr buf, xmlEntityPtr ent);
#endif

xmlChar *xmlEncodeAttributeEntities(xmlDocPtr doc, const xmlChar *input);

#ifdef __cplusplus
}
#endif
#endif /* __XML_SAVE_H__ */

