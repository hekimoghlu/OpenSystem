/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#ifndef ClipboardUtilitiesWin_h
#define ClipboardUtilitiesWin_h

#include "DragData.h"
#include <windows.h>
#include <wtf/Forward.h>

namespace WebCore {

class Document;
class DocumentFragment;

HGLOBAL createGlobalData(const String&);
HGLOBAL createGlobalData(const Vector<char>&);
HGLOBAL createGlobalData(const URL& url, const String& title);
HGLOBAL createGlobalData(std::span<const uint8_t>);

FORMATETC* urlWFormat();
FORMATETC* urlFormat();
FORMATETC* plainTextWFormat();
FORMATETC* plainTextFormat();
FORMATETC* filenameWFormat();
FORMATETC* filenameFormat();
FORMATETC* htmlFormat();
FORMATETC* cfHDropFormat();
FORMATETC* smartPasteFormat();
FORMATETC* fileDescriptorFormat();
FORMATETC* fileContentFormatZero();

void markupToCFHTML(const String& markup, const String& srcURL, Vector<char>& result);

void replaceNewlinesWithWindowsStyleNewlines(String&);
void replaceNBSPWithSpace(String&);

bool containsFilenames(const IDataObject*);
bool containsFilenames(const DragDataMap*);
bool containsHTML(IDataObject*);
bool containsHTML(const DragDataMap*);

RefPtr<DocumentFragment> fragmentFromFilenames(Document*, const IDataObject*);
RefPtr<DocumentFragment> fragmentFromFilenames(Document*, const DragDataMap*);
RefPtr<DocumentFragment> fragmentFromHTML(Document*, IDataObject*);
RefPtr<DocumentFragment> fragmentFromHTML(Document*, const DragDataMap*);
Ref<DocumentFragment> fragmentFromCFHTML(Document*, const String& cfhtml);

String getURL(IDataObject*, DragData::FilenameConversionPolicy, String* title = 0);
String getURL(const DragDataMap*, DragData::FilenameConversionPolicy, String* title = 0);
String getPlainText(IDataObject*);
String getPlainText(const DragDataMap*);
String getTextHTML(IDataObject*);
String getTextHTML(const DragDataMap*);
String getCFHTML(IDataObject*);
String getCFHTML(const DragDataMap*);

void getClipboardData(IDataObject*, FORMATETC* fetc, Vector<String>& dataStrings);
void setClipboardData(IDataObject*, UINT format, const Vector<String>& dataStrings);
void getFileDescriptorData(IDataObject*, int& size, String& pathname);
void getFileContentData(IDataObject*, int size, void* dataBlob);
void setFileDescriptorData(IDataObject*, int size, const String& pathname);
void setFileContentData(IDataObject*, int size, void* dataBlob);

} // namespace WebCore

#endif // ClipboardUtilitiesWin_h
