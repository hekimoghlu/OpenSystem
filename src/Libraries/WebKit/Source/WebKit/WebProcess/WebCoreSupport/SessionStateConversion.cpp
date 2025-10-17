/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#include "config.h"
#include "SessionStateConversion.h"

#include "SessionState.h"
#include <WebCore/BlobData.h>
#include <WebCore/FormData.h>
#include <WebCore/HistoryItem.h>
#include <wtf/FileSystem.h>

namespace WebKit {
using namespace WebCore;

static HTTPBody toHTTPBody(const FormData& formData)
{
    HTTPBody httpBody;

    for (const auto& formDataElement : formData.elements()) {
        HTTPBody::Element element;

        switchOn(formDataElement.data,
            [&] (const Vector<uint8_t>& bytes) {
                element.data = bytes;
            }, [&] (const FormDataElement::EncodedFileData& fileData) {
                HTTPBody::Element::FileData data;
                data.filePath = fileData.filename;
                data.fileStart = fileData.fileStart;
                if (fileData.fileLength != BlobDataItem::toEndOfFile)
                    data.fileLength = fileData.fileLength;
                data.expectedFileModificationTime = fileData.expectedFileModificationTime;
                element.data = WTFMove(data);
            }, [&] (const FormDataElement::EncodedBlobData& blobData) {
                element.data = blobData.url.string();
            }
        );

        httpBody.elements.append(WTFMove(element));
    }

    return httpBody;
}

Ref<FrameState> toFrameState(const HistoryItem& historyItem)
{
    Ref frameState = FrameState::create();

    frameState->urlString = historyItem.urlString();
    frameState->originalURLString = historyItem.originalURLString();
    frameState->referrer = historyItem.referrer();
    frameState->target = historyItem.target();
    frameState->frameID = historyItem.frameID();

    frameState->setDocumentState(historyItem.documentState());
    if (RefPtr<SerializedScriptValue> stateObject = historyItem.stateObject())
        frameState->stateObjectData = stateObject->wireBytes();

    frameState->documentSequenceNumber = historyItem.documentSequenceNumber();
    frameState->itemSequenceNumber = historyItem.itemSequenceNumber();

    frameState->scrollPosition = historyItem.scrollPosition();
    frameState->shouldRestoreScrollPosition = historyItem.shouldRestoreScrollPosition();
    frameState->pageScaleFactor = historyItem.pageScaleFactor();

    if (FormData* formData = const_cast<HistoryItem&>(historyItem).formData()) {
        HTTPBody httpBody = toHTTPBody(*formData);
        httpBody.contentType = historyItem.formContentType();

        frameState->httpBody = WTFMove(httpBody);
    }

    frameState->itemID = historyItem.itemID();
    frameState->frameItemID = historyItem.frameItemID();
    frameState->hasCachedPage = historyItem.isInBackForwardCache();
    frameState->shouldOpenExternalURLsPolicy = historyItem.shouldOpenExternalURLsPolicy();
    frameState->sessionStateObject = historyItem.stateObject();
    frameState->wasCreatedByJSWithoutUserInteraction = historyItem.wasCreatedByJSWithoutUserInteraction();
    frameState->wasRestoredFromSession = historyItem.wasRestoredFromSession();
    frameState->policyContainer = historyItem.policyContainer();

    static constexpr auto maxTitleLength = 1000u; // Closest power of 10 above the W3C recommendation for Title length.
    frameState->title = historyItem.title().left(maxTitleLength);

#if PLATFORM(IOS_FAMILY)
    frameState->exposedContentRect = historyItem.exposedContentRect();
    frameState->unobscuredContentRect = historyItem.unobscuredContentRect();
    frameState->minimumLayoutSizeInScrollViewCoordinates = historyItem.minimumLayoutSizeInScrollViewCoordinates();
    frameState->contentSize = historyItem.contentSize();
    frameState->scaleIsInitial = historyItem.scaleIsInitial();
    frameState->obscuredInsets = historyItem.obscuredInsets();
#endif

    frameState->children = historyItem.children().map([](auto& childHistoryItem) {
        return toFrameState(childHistoryItem);
    });

    return frameState;
}

static Ref<FormData> toFormData(const HTTPBody& httpBody)
{
    auto formData = FormData::create();

    for (const auto& element : httpBody.elements) {
        switchOn(element.data, [&] (const Vector<uint8_t>& data) {
            formData->appendData(data.span());
        }, [&] (const HTTPBody::Element::FileData& data) {
            formData->appendFileRange(data.filePath, data.fileStart, data.fileLength.value_or(BlobDataItem::toEndOfFile), data.expectedFileModificationTime);
        }, [&] (const String& blobURLString) {
            formData->appendBlob(URL { blobURLString });
        });
    }

    return formData;
}

static void applyFrameState(HistoryItemClient& client, HistoryItem& historyItem, const FrameState& frameState)
{
    historyItem.setOriginalURLString(frameState.originalURLString);
    historyItem.setReferrer(frameState.referrer);
    historyItem.setTarget(frameState.target);
    historyItem.setFrameID(frameState.frameID);

    historyItem.setDocumentState(frameState.documentState());

    if (frameState.stateObjectData) {
        Vector<uint8_t> stateObjectData = frameState.stateObjectData.value();
        historyItem.setStateObject(SerializedScriptValue::createFromWireBytes(WTFMove(stateObjectData)));
    }

    historyItem.setDocumentSequenceNumber(frameState.documentSequenceNumber);
    historyItem.setItemSequenceNumber(frameState.itemSequenceNumber);

    historyItem.setScrollPosition(frameState.scrollPosition);
    historyItem.setShouldRestoreScrollPosition(frameState.shouldRestoreScrollPosition);
    historyItem.setPageScaleFactor(frameState.pageScaleFactor);

    if (frameState.httpBody) {
        const auto& httpBody = frameState.httpBody.value();
        historyItem.setFormContentType(httpBody.contentType);

        historyItem.setFormData(toFormData(httpBody));
    }

    historyItem.setShouldOpenExternalURLsPolicy(frameState.shouldOpenExternalURLsPolicy);
    historyItem.setStateObject(frameState.sessionStateObject.get());
    historyItem.setWasCreatedByJSWithoutUserInteraction(frameState.wasCreatedByJSWithoutUserInteraction);
    historyItem.setWasRestoredFromSession(frameState.wasRestoredFromSession);
    if (auto policyContainer = frameState.policyContainer)
        historyItem.setPolicyContainer(*policyContainer);

#if PLATFORM(IOS_FAMILY)
    historyItem.setExposedContentRect(frameState.exposedContentRect);
    historyItem.setUnobscuredContentRect(frameState.unobscuredContentRect);
    historyItem.setMinimumLayoutSizeInScrollViewCoordinates(frameState.minimumLayoutSizeInScrollViewCoordinates);
    historyItem.setContentSize(frameState.contentSize);
    historyItem.setScaleIsInitial(frameState.scaleIsInitial);
    historyItem.setObscuredInsets(frameState.obscuredInsets);
#endif

    for (auto& childFrameState : frameState.children) {
        Ref childHistoryItem = HistoryItem::create(client, childFrameState->urlString, { }, { }, childFrameState->itemID, childFrameState->frameItemID);
        applyFrameState(client, childHistoryItem, childFrameState);

        historyItem.addChildItem(WTFMove(childHistoryItem));
    }
}

Ref<HistoryItem> toHistoryItem(HistoryItemClient& client, const FrameState& frameState)
{
    Ref historyItem = HistoryItem::create(client, frameState.urlString, frameState.title, { }, frameState.itemID, frameState.frameItemID);
    applyFrameState(client, historyItem, frameState);
    return historyItem;
}

} // namespace WebKit
