/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
#import <Foundation/Foundation.h>

@class DOMElement;
@class WebArchive;
@class WebHTMLView;

#ifdef __cplusplus
extern "C" {
#endif

extern NSString *WebURLPboardType;
extern NSString *WebURLNamePboardType;

@interface NSPasteboard (WebExtras)

// Returns the array of types that _web_writeURL:andTitle: handles.
+ (NSArray *)_web_writableTypesForURL;

// Returns the array of types that _web_writeImage handles.
+ (NSArray *)_web_writableTypesForImageIncludingArchive:(BOOL)hasArchive;

// Returns the array of drag types that _web_bestURL handles; note that the presence
// of one or more of these drag types on the pasteboard is not a guarantee that
// _web_bestURL will return a non-nil result.
+ (NSArray *)_web_dragTypesForURL;

// Finds the best URL from the data on the pasteboard, giving priority to http and https URLs
- (NSURL *)_web_bestURL;

// Writes the URL to the pasteboard with the passed types.
- (void)_web_writeURL:(NSURL *)URL andTitle:(NSString *)title types:(NSArray *)types;

// Sets the text on the find pasteboard (NSPasteboardNameFind). Returns the new changeCount for the find pasteboard.
+ (int)_web_setFindPasteboardString:(NSString *)string withOwner:(id)owner;

// Writes a file wrapper to the pasteboard as an RTFD attachment.
// NSRTFDPboardType must be declared on the pasteboard before calling this method.
- (void)_web_writeFileWrapperAsRTFDAttachment:(NSFileWrapper *)wrapper;

// Writes an image, URL and other optional types to the pasteboard.
- (void)_web_writeImage:(NSImage *)image 
                element:(DOMElement*)element
                    URL:(NSURL *)URL 
                  title:(NSString *)title
                archive:(WebArchive *)archive
                  types:(NSArray *)types
                 source:(WebHTMLView *)source;

- (id)_web_declareAndWriteDragImageForElement:(DOMElement *)element
                                       URL:(NSURL *)URL 
                                     title:(NSString *)title
                                   archive:(WebArchive *)archive
                                    source:(WebHTMLView *)source;

- (void)_web_writePromisedRTFDFromArchive:(WebArchive*)archive containsImage:(BOOL)containsImage;

@end

#ifdef __cplusplus
}
#endif
