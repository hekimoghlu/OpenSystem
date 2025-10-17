/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
#import "WebPDFDocumentExtras.h"

#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <wtf/Vector.h>
#import <wtf/RetainPtr.h>

#if !PLATFORM(IOS_FAMILY)
#import <Quartz/Quartz.h>
#endif

static void appendValuesInPDFNameSubtreeToVector(CGPDFDictionaryRef subtree, Vector<CGPDFObjectRef>& values)
{
    CGPDFArrayRef names;
    if (CGPDFDictionaryGetArray(subtree, "Names", &names)) {
        size_t nameCount = CGPDFArrayGetCount(names) / 2;
        for (size_t i = 0; i < nameCount; ++i) {
            CGPDFObjectRef object;
            CGPDFArrayGetObject(names, 2 * i + 1, &object);
            values.append(object);
        }
        return;
    }

    CGPDFArrayRef kids;
    if (!CGPDFDictionaryGetArray(subtree, "Kids", &kids))
        return;

    size_t kidCount = CGPDFArrayGetCount(kids);
    for (size_t i = 0; i < kidCount; ++i) {
        CGPDFDictionaryRef kid;
        if (!CGPDFArrayGetDictionary(kids, i, &kid))
            continue;
        appendValuesInPDFNameSubtreeToVector(kid, values);
    }
}

static void getAllValuesInPDFNameTree(CGPDFDictionaryRef tree, Vector<CGPDFObjectRef>& allValues)
{
    appendValuesInPDFNameSubtreeToVector(tree, allValues);
}

NSArray *allScriptsInPDFDocument(CGPDFDocumentRef pdfDocument)
{
    NSMutableArray *scripts = [NSMutableArray array];
    if (!pdfDocument)
        return scripts;

    CGPDFDictionaryRef pdfCatalog = CGPDFDocumentGetCatalog(pdfDocument);
    if (!pdfCatalog)
        return scripts;

    // Get the dictionary of all document-level name trees.
    CGPDFDictionaryRef namesDictionary;
    if (!CGPDFDictionaryGetDictionary(pdfCatalog, "Names", &namesDictionary))
        return scripts;

    // Get the document-level "JavaScript" name tree.
    CGPDFDictionaryRef javaScriptNameTree;
    if (!CGPDFDictionaryGetDictionary(namesDictionary, "JavaScript", &javaScriptNameTree))
        return scripts;

    // The names are arbitrary. We are only interested in the values.
    Vector<CGPDFObjectRef> objects;
    getAllValuesInPDFNameTree(javaScriptNameTree, objects);
    size_t objectCount = objects.size();

    for (size_t i = 0; i < objectCount; ++i) {
        CGPDFDictionaryRef javaScriptAction;
        if (!CGPDFObjectGetValue(reinterpret_cast<CGPDFObjectRef>(objects[i]), kCGPDFObjectTypeDictionary, &javaScriptAction))
            continue;

        // A JavaScript action must have an action type of "JavaScript".
        if (CGPDFDictionaryGetNameString(javaScriptAction, "S"_s) != "JavaScript"_s)
            continue;

        const UInt8* bytes = nullptr;
        CFIndex length = 0;
        CGPDFStreamRef stream;
        CGPDFStringRef string;
        RetainPtr<CFDataRef> data;
        if (CGPDFDictionaryGetStream(javaScriptAction, "JS", &stream)) {
            CGPDFDataFormat format;
            data = adoptCF(CGPDFStreamCopyData(stream, &format));
            if (!data)
                continue;
            bytes = CFDataGetBytePtr(data.get());
            length = CFDataGetLength(data.get());
        } else if (CGPDFDictionaryGetString(javaScriptAction, "JS", &string)) {
            bytes = CGPDFStringGetBytePtr(string);
            length = CGPDFStringGetLength(string);
        }
        if (!bytes || !length)
            continue;

        NSStringEncoding encoding = (length > 1 && bytes[0] == 0xFE && bytes[1] == 0xFF) ? NSUnicodeStringEncoding : NSUTF8StringEncoding;
        auto script = adoptNS([[NSString alloc] initWithBytes:bytes length:length encoding:encoding]);
        if (!script)
            continue;

        [scripts addObject:script.get()];
    }

    return scripts;
}

#if !PLATFORM(IOS_FAMILY)
NSArray *allScriptsInPDFDocument(PDFDocument *document)
{
    return allScriptsInPDFDocument([document documentRef]);
}
#endif
