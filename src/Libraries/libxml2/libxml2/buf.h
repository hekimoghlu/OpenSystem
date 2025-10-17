/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#ifndef __XML_BUF_H__
#define __XML_BUF_H__

#include <libxml/tree.h>

#ifdef __cplusplus
extern "C" {
#endif

xmlBufPtr xmlBufCreate(void);
xmlBufPtr xmlBufCreateSize(size_t size);
xmlBufPtr xmlBufCreateStatic(void *mem, size_t size);

int xmlBufSetAllocationScheme(xmlBufPtr buf,
                              xmlBufferAllocationScheme scheme);
int xmlBufGetAllocationScheme(xmlBufPtr buf);

void xmlBufFree(xmlBufPtr buf);
void xmlBufEmpty(xmlBufPtr buf);

/* size_t xmlBufShrink(xmlBufPtr buf, size_t len); */
int xmlBufGrow(xmlBufPtr buf, int len);
int xmlBufInflate(xmlBufPtr buf, size_t len);
int xmlBufResize(xmlBufPtr buf, size_t len);

int xmlBufAdd(xmlBufPtr buf, const xmlChar *str, int len);
int xmlBufAddHead(xmlBufPtr buf, const xmlChar *str, int len);
int xmlBufCat(xmlBufPtr buf, const xmlChar *str);
int xmlBufCCat(xmlBufPtr buf, const char *str);
int xmlBufWriteCHAR(xmlBufPtr buf, const xmlChar *string);
int xmlBufWriteChar(xmlBufPtr buf, const char *string);
int xmlBufWriteQuotedString(xmlBufPtr buf, const xmlChar *string);

size_t xmlBufAvail(const xmlBufPtr buf);
size_t xmlBufLength(const xmlBufPtr buf);
/* size_t xmlBufUse(const xmlBufPtr buf); */
int xmlBufIsEmpty(const xmlBufPtr buf);
int xmlBufAddLen(xmlBufPtr buf, size_t len);
int xmlBufErase(xmlBufPtr buf, size_t len);

/* const xmlChar * xmlBufContent(const xmlBuf *buf); */
/* const xmlChar * xmlBufEnd(xmlBufPtr buf); */

xmlChar * xmlBufDetach(xmlBufPtr buf);

size_t xmlBufDump(FILE *file, xmlBufPtr buf);

xmlBufPtr xmlBufFromBuffer(xmlBufferPtr buffer);
xmlBufferPtr xmlBufBackToBuffer(xmlBufPtr buf);
int xmlBufMergeBuffer(xmlBufPtr buf, xmlBufferPtr buffer);

int xmlBufResetInput(xmlBufPtr buf, xmlParserInputPtr input);
size_t xmlBufGetInputBase(xmlBufPtr buf, xmlParserInputPtr input);
int xmlBufSetInputBaseCur(xmlBufPtr buf, xmlParserInputPtr input,
                          size_t base, size_t cur);
#ifdef __cplusplus
}
#endif
#endif /* __XML_BUF_H__ */

