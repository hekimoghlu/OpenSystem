/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#if !PLATFORM(IOS_FAMILY)

#import "WebPDFRepresentation.h"

#import "WebDataSourcePrivate.h"
#import "WebFrame.h"
#import "WebJSPDFDoc.h"
#import "WebPDFDocumentExtras.h"
#import "WebPDFView.h"
#import <JavaScriptCore/JSContextRef.h>
#import <JavaScriptCore/OpaqueJSString.h>
#import <wtf/Assertions.h>
#import <wtf/RetainPtr.h>

@implementation WebPDFRepresentation

+ (NSArray *)supportedMIMETypes
{
    return @[ @"text/pdf", @"application/pdf" ];
}

+ (Class)PDFDocumentClass
{
    static Class PDFDocumentClass = nil;
    if (PDFDocumentClass == nil) {
        PDFDocumentClass = [[WebPDFView PDFKitBundle] classNamed:@"PDFDocument"];
        if (PDFDocumentClass == nil) {
            LOG_ERROR("Couldn't find PDFDocument class in PDFKit.framework");
        }
    }
    return PDFDocumentClass;
}

- (void)setDataSource:(WebDataSource *)dataSource
{
}

- (void)receivedData:(NSData *)data withDataSource:(WebDataSource *)dataSource
{
}

- (void)receivedError:(NSError *)error withDataSource:(WebDataSource *)dataSource
{
}

- (void)finishedLoadingWithDataSource:(WebDataSource *)dataSource
{
    WebPDFView *view = (WebPDFView *)[[[dataSource webFrame] frameView] documentView];
    auto document = adoptNS([[[[self class] PDFDocumentClass] alloc] initWithData:[dataSource data]]);
    [view setPDFDocument:document.get()];

    NSArray *scripts = allScriptsInPDFDocument(document.get());
    if (![scripts count])
        return;

    JSGlobalContextRef ctx = JSGlobalContextCreate(0);
    JSObjectRef jsPDFDoc = makeJSPDFDoc(ctx, dataSource);
    for (NSString *script in scripts)
        JSEvaluateScript(ctx, OpaqueJSString::tryCreate(script).get(), jsPDFDoc, nullptr, 0, nullptr);
    JSGlobalContextRelease(ctx);
}

- (BOOL)canProvideDocumentSource
{
    return NO;
}


- (NSString *)documentSource
{
    return nil;
}


- (NSString *)title
{
    return nil;
}

@end

#endif // !PLATFORM(IOS_FAMILY)
