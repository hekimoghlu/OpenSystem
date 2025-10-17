/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
#pragma once

#include <WebKit/WKBase.h>
#include <WebKit/WKEvent.h>

enum {
    WKBundlePageUIElementVisibilityUnknown,
    WKBundlePageUIElementVisible,
    WKBundlePageUIElementHidden
};
typedef uint32_t WKBundlePageUIElementVisibility;


typedef void (*WKBundlePageWillAddMessageToConsoleCallback)(WKBundlePageRef page, WKStringRef message, uint32_t lineNumber, const void *clientInfo);
typedef void (*WKBundlePageWillSetStatusbarTextCallback)(WKBundlePageRef page, WKStringRef statusbarText, const void *clientInfo);
typedef void (*WKBundlePageWillRunJavaScriptAlertCallback)(WKBundlePageRef page, WKStringRef alertText, WKBundleFrameRef frame, const void *clientInfo);
typedef void (*WKBundlePageWillRunJavaScriptConfirmCallback)(WKBundlePageRef page, WKStringRef message, WKBundleFrameRef frame, const void *clientInfo);
typedef void (*WKBundlePageWillRunJavaScriptPromptCallback)(WKBundlePageRef page, WKStringRef message, WKStringRef defaultValue, WKBundleFrameRef frame, const void *clientInfo);
typedef void (*WKBundlePageMouseDidMoveOverElementCallback)(WKBundlePageRef page, WKBundleHitTestResultRef hitTestResult, WKEventModifiers modifiers, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidScrollCallback)(WKBundlePageRef page, const void *clientInfo);
typedef WKStringRef (*WKBundlePageGenerateFileForUploadCallback)(WKBundlePageRef page, WKStringRef originalFilePath, const void* clientInfo);
typedef WKBundlePageUIElementVisibility (*WKBundlePageStatusBarIsVisibleCallback)(WKBundlePageRef page, const void *clientInfo);
typedef WKBundlePageUIElementVisibility (*WKBundlePageMenuBarIsVisibleCallback)(WKBundlePageRef page, const void *clientInfo);
typedef WKBundlePageUIElementVisibility (*WKBundlePageToolbarsAreVisibleCallback)(WKBundlePageRef page, const void *clientInfo);
typedef void (*WKBundlePageReachedAppCacheOriginQuotaCallback)(WKBundlePageRef page, WKSecurityOriginRef origin, int64_t totalBytesNeeded, const void *clientInfo);
typedef uint64_t (*WKBundlePageExceededDatabaseQuotaCallback)(WKBundlePageRef page, WKSecurityOriginRef origin, WKStringRef databaseName, WKStringRef databaseDisplayName, uint64_t currentQuotaBytes, uint64_t currentOriginUsageBytes, uint64_t currentDatabaseUsageBytes, uint64_t expectedUsageBytes, const void *clientInfo);
typedef WKStringRef (*WKBundlePagePlugInCreateStartLabelTitleCallback)(WKStringRef mimeType, const void *clientInfo);
typedef WKStringRef (*WKBundlePagePlugInCreateStartLabelSubtitleCallback)(WKStringRef mimeType, const void *clientInfo);
typedef WKStringRef (*WKBundlePagePlugInCreateExtraStyleSheetCallback)(const void *clientInfo);
typedef WKStringRef (*WKBundlePagePlugInCreateExtraScriptCallback)(const void *clientInfo);
typedef void (*WKBundlePageDidClickAutoFillButtonCallback)(WKBundlePageRef page, WKBundleNodeHandleRef inputElement, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageDidResignInputElementStrongPasswordAppearance)(WKBundlePageRef page, WKBundleNodeHandleRef inputElement, WKTypeRef* userData, const void *clientInfo);
typedef void (*WKBundlePageWillAddMessageWithDetailsToConsoleCallback)(WKBundlePageRef page, WKStringRef message, WKArrayRef messageArguments, uint32_t lineNumber, uint32_t columnNumber, WKStringRef sourceID, const void *clientInfo);

typedef struct WKBundlePageUIClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKBundlePageUIClientBase;

typedef struct WKBundlePageUIClientV0 {
    WKBundlePageUIClientBase                                            base;

    // Version 0.
    WKBundlePageWillAddMessageToConsoleCallback                         willAddMessageToConsole;
    WKBundlePageWillSetStatusbarTextCallback                            willSetStatusbarText;
    WKBundlePageWillRunJavaScriptAlertCallback                          willRunJavaScriptAlert;
    WKBundlePageWillRunJavaScriptConfirmCallback                        willRunJavaScriptConfirm;
    WKBundlePageWillRunJavaScriptPromptCallback                         willRunJavaScriptPrompt;
    WKBundlePageMouseDidMoveOverElementCallback                         mouseDidMoveOverElement;
    WKBundlePageDidScrollCallback                                       pageDidScroll;
    void*                                                               unused1;
    WKBundlePageGenerateFileForUploadCallback                           shouldGenerateFileForUpload;
    WKBundlePageGenerateFileForUploadCallback                           generateFileForUpload;
    void*                                                               unused2;
    WKBundlePageStatusBarIsVisibleCallback                              statusBarIsVisible;
    WKBundlePageMenuBarIsVisibleCallback                                menuBarIsVisible;
    WKBundlePageToolbarsAreVisibleCallback                              toolbarsAreVisible;
} WKBundlePageUIClientV0;

typedef struct WKBundlePageUIClientV1 {
    WKBundlePageUIClientBase                                            base;

    // Version 0.
    WKBundlePageWillAddMessageToConsoleCallback                         willAddMessageToConsole;
    WKBundlePageWillSetStatusbarTextCallback                            willSetStatusbarText;
    WKBundlePageWillRunJavaScriptAlertCallback                          willRunJavaScriptAlert;
    WKBundlePageWillRunJavaScriptConfirmCallback                        willRunJavaScriptConfirm;
    WKBundlePageWillRunJavaScriptPromptCallback                         willRunJavaScriptPrompt;
    WKBundlePageMouseDidMoveOverElementCallback                         mouseDidMoveOverElement;
    WKBundlePageDidScrollCallback                                       pageDidScroll;
    void*                                                               unused1;
    WKBundlePageGenerateFileForUploadCallback                           shouldGenerateFileForUpload;
    WKBundlePageGenerateFileForUploadCallback                           generateFileForUpload;
    void*                                                               unused2;
    WKBundlePageStatusBarIsVisibleCallback                              statusBarIsVisible;
    WKBundlePageMenuBarIsVisibleCallback                                menuBarIsVisible;
    WKBundlePageToolbarsAreVisibleCallback                              toolbarsAreVisible;

    // Version 1.
    WKBundlePageReachedAppCacheOriginQuotaCallback                      didReachApplicationCacheOriginQuota;
} WKBundlePageUIClientV1;

typedef struct WKBundlePageUIClientV2 {
    WKBundlePageUIClientBase                                            base;

    // Version 0.
    WKBundlePageWillAddMessageToConsoleCallback                         willAddMessageToConsole;
    WKBundlePageWillSetStatusbarTextCallback                            willSetStatusbarText;
    WKBundlePageWillRunJavaScriptAlertCallback                          willRunJavaScriptAlert;
    WKBundlePageWillRunJavaScriptConfirmCallback                        willRunJavaScriptConfirm;
    WKBundlePageWillRunJavaScriptPromptCallback                         willRunJavaScriptPrompt;
    WKBundlePageMouseDidMoveOverElementCallback                         mouseDidMoveOverElement;
    WKBundlePageDidScrollCallback                                       pageDidScroll;
    void*                                                               unused1;
    WKBundlePageGenerateFileForUploadCallback                           shouldGenerateFileForUpload;
    WKBundlePageGenerateFileForUploadCallback                           generateFileForUpload;
    void*                                                               unused2;
    WKBundlePageStatusBarIsVisibleCallback                              statusBarIsVisible;
    WKBundlePageMenuBarIsVisibleCallback                                menuBarIsVisible;
    WKBundlePageToolbarsAreVisibleCallback                              toolbarsAreVisible;

    // Version 1.
    WKBundlePageReachedAppCacheOriginQuotaCallback                      didReachApplicationCacheOriginQuota;

    // Version 2.
    WKBundlePageExceededDatabaseQuotaCallback                           didExceedDatabaseQuota;
    WKBundlePagePlugInCreateStartLabelTitleCallback                     createPlugInStartLabelTitle;
    WKBundlePagePlugInCreateStartLabelSubtitleCallback                  createPlugInStartLabelSubtitle;
    WKBundlePagePlugInCreateExtraStyleSheetCallback                     createPlugInExtraStyleSheet;
    WKBundlePagePlugInCreateExtraScriptCallback                         createPlugInExtraScript;
} WKBundlePageUIClientV2;

typedef struct WKBundlePageUIClientV3 {
    WKBundlePageUIClientBase                                            base;

    // Version 0.
    WKBundlePageWillAddMessageToConsoleCallback                         willAddMessageToConsole;
    WKBundlePageWillSetStatusbarTextCallback                            willSetStatusbarText;
    WKBundlePageWillRunJavaScriptAlertCallback                          willRunJavaScriptAlert;
    WKBundlePageWillRunJavaScriptConfirmCallback                        willRunJavaScriptConfirm;
    WKBundlePageWillRunJavaScriptPromptCallback                         willRunJavaScriptPrompt;
    WKBundlePageMouseDidMoveOverElementCallback                         mouseDidMoveOverElement;
    WKBundlePageDidScrollCallback                                       pageDidScroll;
    void*                                                               unused1;
    WKBundlePageGenerateFileForUploadCallback                           shouldGenerateFileForUpload;
    WKBundlePageGenerateFileForUploadCallback                           generateFileForUpload;
    void*                                                               unused2;
    WKBundlePageStatusBarIsVisibleCallback                              statusBarIsVisible;
    WKBundlePageMenuBarIsVisibleCallback                                menuBarIsVisible;
    WKBundlePageToolbarsAreVisibleCallback                              toolbarsAreVisible;

    // Version 1.
    WKBundlePageReachedAppCacheOriginQuotaCallback                      didReachApplicationCacheOriginQuota;

    // Version 2.
    WKBundlePageExceededDatabaseQuotaCallback                           didExceedDatabaseQuota;
    WKBundlePagePlugInCreateStartLabelTitleCallback                     createPlugInStartLabelTitle;
    WKBundlePagePlugInCreateStartLabelSubtitleCallback                  createPlugInStartLabelSubtitle;
    WKBundlePagePlugInCreateExtraStyleSheetCallback                     createPlugInExtraStyleSheet;
    WKBundlePagePlugInCreateExtraScriptCallback                         createPlugInExtraScript;

    // Version 3.
    void*                                                               unused3;
    void*                                                               unused4;
    void*                                                               unused5;

    WKBundlePageDidClickAutoFillButtonCallback                          didClickAutoFillButton;
} WKBundlePageUIClientV3;

typedef struct WKBundlePageUIClientV4 {
    WKBundlePageUIClientBase                                            base;

    // Version 0.
    WKBundlePageWillAddMessageToConsoleCallback                         willAddMessageToConsole;
    WKBundlePageWillSetStatusbarTextCallback                            willSetStatusbarText;
    WKBundlePageWillRunJavaScriptAlertCallback                          willRunJavaScriptAlert;
    WKBundlePageWillRunJavaScriptConfirmCallback                        willRunJavaScriptConfirm;
    WKBundlePageWillRunJavaScriptPromptCallback                         willRunJavaScriptPrompt;
    WKBundlePageMouseDidMoveOverElementCallback                         mouseDidMoveOverElement;
    WKBundlePageDidScrollCallback                                       pageDidScroll;
    void*                                                               unused1;
    WKBundlePageGenerateFileForUploadCallback                           shouldGenerateFileForUpload;
    WKBundlePageGenerateFileForUploadCallback                           generateFileForUpload;
    void*                                                               unused2;
    WKBundlePageStatusBarIsVisibleCallback                              statusBarIsVisible;
    WKBundlePageMenuBarIsVisibleCallback                                menuBarIsVisible;
    WKBundlePageToolbarsAreVisibleCallback                              toolbarsAreVisible;

    // Version 1.
    WKBundlePageReachedAppCacheOriginQuotaCallback                      didReachApplicationCacheOriginQuota;

    // Version 2.
    WKBundlePageExceededDatabaseQuotaCallback                           didExceedDatabaseQuota;
    WKBundlePagePlugInCreateStartLabelTitleCallback                     createPlugInStartLabelTitle;
    WKBundlePagePlugInCreateStartLabelSubtitleCallback                  createPlugInStartLabelSubtitle;
    WKBundlePagePlugInCreateExtraStyleSheetCallback                     createPlugInExtraStyleSheet;
    WKBundlePagePlugInCreateExtraScriptCallback                         createPlugInExtraScript;

    // Version 3.
    void*                                                               unused3;
    void*                                                               unused4;
    void*                                                               unused5;

    WKBundlePageDidClickAutoFillButtonCallback                          didClickAutoFillButton;

    // Version 4.
    WKBundlePageDidResignInputElementStrongPasswordAppearance           didResignInputElementStrongPasswordAppearance;
} WKBundlePageUIClientV4;

typedef struct WKBundlePageUIClientV5 {
    WKBundlePageUIClientBase                                            base;

    // Version 0.
    WKBundlePageWillAddMessageToConsoleCallback                         willAddMessageToConsole;
    WKBundlePageWillSetStatusbarTextCallback                            willSetStatusbarText;
    WKBundlePageWillRunJavaScriptAlertCallback                          willRunJavaScriptAlert;
    WKBundlePageWillRunJavaScriptConfirmCallback                        willRunJavaScriptConfirm;
    WKBundlePageWillRunJavaScriptPromptCallback                         willRunJavaScriptPrompt;
    WKBundlePageMouseDidMoveOverElementCallback                         mouseDidMoveOverElement;
    WKBundlePageDidScrollCallback                                       pageDidScroll;
    void*                                                               unused1;
    WKBundlePageGenerateFileForUploadCallback                           shouldGenerateFileForUpload;
    WKBundlePageGenerateFileForUploadCallback                           generateFileForUpload;
    void*                                                               unused2;
    WKBundlePageStatusBarIsVisibleCallback                              statusBarIsVisible;
    WKBundlePageMenuBarIsVisibleCallback                                menuBarIsVisible;
    WKBundlePageToolbarsAreVisibleCallback                              toolbarsAreVisible;

    // Version 1.
    WKBundlePageReachedAppCacheOriginQuotaCallback                      didReachApplicationCacheOriginQuota;

    // Version 2.
    WKBundlePageExceededDatabaseQuotaCallback                           didExceedDatabaseQuota;
    WKBundlePagePlugInCreateStartLabelTitleCallback                     createPlugInStartLabelTitle;
    WKBundlePagePlugInCreateStartLabelSubtitleCallback                  createPlugInStartLabelSubtitle;
    WKBundlePagePlugInCreateExtraStyleSheetCallback                     createPlugInExtraStyleSheet;
    WKBundlePagePlugInCreateExtraScriptCallback                         createPlugInExtraScript;

    // Version 3.
    void*                                                               unused3;
    void*                                                               unused4;
    void*                                                               unused5;

    WKBundlePageDidClickAutoFillButtonCallback                          didClickAutoFillButton;

    // Version 4.
    WKBundlePageDidResignInputElementStrongPasswordAppearance           didResignInputElementStrongPasswordAppearance;

    // Version 5.
    WKBundlePageWillAddMessageWithDetailsToConsoleCallback              willAddMessageWithDetailsToConsole;
} WKBundlePageUIClientV5;
