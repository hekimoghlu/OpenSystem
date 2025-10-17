/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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

#include "BackForwardItemIdentifier.h"
#include "FrameLoader.h"
#include <wtf/CheckedPtr.h>

namespace WebCore {

class HistoryItem;
class HistoryItemClient;
class LocalFrame;
class SerializedScriptValue;

enum class ShouldTreatAsContinuingLoad : uint8_t;

struct NavigationAPIMethodTracker;
struct StringWithDirection;

class HistoryController final : public CanMakeCheckedPtr<HistoryController>, public CanMakeWeakPtr<HistoryController>  {
    WTF_MAKE_NONCOPYABLE(HistoryController);
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HistoryController);
public:
    enum HistoryUpdateType { UpdateAll, UpdateAllExceptBackForwardList };

    explicit HistoryController(LocalFrame&);
    ~HistoryController();

    void ref() const;
    void deref() const;

    WEBCORE_EXPORT void saveScrollPositionAndViewStateToItem(HistoryItem*);
    WEBCORE_EXPORT void restoreScrollPositionAndViewState();

    void updateBackForwardListForFragmentScroll();

    void saveDocumentState();
    WEBCORE_EXPORT void saveDocumentAndScrollState();
    void restoreDocumentState();

    void invalidateCurrentItemCachedPage();

    void updateForBackForwardNavigation();
    void updateForReload();
    void updateForStandardLoad(HistoryUpdateType updateType = UpdateAll);
    void updateForRedirectWithLockedBackForwardList();
    void updateForClientRedirect();
    void updateForCommit();
    void updateForSameDocumentNavigation();
    void updateForFrameLoadCompleted();

    HistoryItem* currentItem() const { return m_currentItem.get(); }
    RefPtr<HistoryItem> protectedCurrentItem() const;
    WEBCORE_EXPORT void setCurrentItem(Ref<HistoryItem>&&);
    void setCurrentItemTitle(const StringWithDirection&);
    bool currentItemShouldBeReplaced() const;
    WEBCORE_EXPORT void replaceCurrentItem(RefPtr<HistoryItem>&&);

    HistoryItem* previousItem() const { return m_previousItem.get(); }
    RefPtr<HistoryItem> protectedPreviousItem() const;
    void clearPreviousItem();

    HistoryItem* provisionalItem() const { return m_provisionalItem.get(); }
    RefPtr<HistoryItem> protectedProvisionalItem() const;
    void setProvisionalItem(RefPtr<HistoryItem>&&);

    void pushState(RefPtr<SerializedScriptValue>&&, const String& url);
    void replaceState(RefPtr<SerializedScriptValue>&&, const String& url);

    void setDefersLoading(bool);

    Ref<HistoryItem> createItemWithLoader(HistoryItemClient&, DocumentLoader*);

    WEBCORE_EXPORT RefPtr<HistoryItem> createItemTree(LocalFrame& targetFrame, bool clipAtTarget, BackForwardItemIdentifier);

    void clearPolicyItem();

private:
    friend class Page;
    bool shouldStopLoadingForHistoryItem(HistoryItem&) const;
    void goToItem(HistoryItem&, FrameLoadType, ShouldTreatAsContinuingLoad);
    void goToItemForNavigationAPI(HistoryItem&, FrameLoadType, LocalFrame& triggeringFrame, NavigationAPIMethodTracker*);
    void goToItemShared(HistoryItem&, CompletionHandler<void(bool)>&&);

    void initializeItem(HistoryItem&, RefPtr<DocumentLoader>);
    Ref<HistoryItem> createItem(HistoryItemClient&, BackForwardItemIdentifier);
    Ref<HistoryItem> createItemTree(HistoryItemClient&, LocalFrame& targetFrame, bool clipAtTarget, BackForwardItemIdentifier);

    enum class ForNavigationAPI : bool { No, Yes };
    void recursiveSetProvisionalItem(HistoryItem&, HistoryItem*, ForNavigationAPI = ForNavigationAPI::No);
    void recursiveGoToItem(HistoryItem&, HistoryItem*, FrameLoadType, ShouldTreatAsContinuingLoad);
    bool isReplaceLoadTypeWithProvisionalItem(FrameLoadType);
    bool isReloadTypeWithProvisionalItem(FrameLoadType);
    void recursiveUpdateForCommit();
    void recursiveUpdateForSameDocumentNavigation();
    static bool itemsAreClones(HistoryItem&, HistoryItem*);
    void updateBackForwardListClippedAtTarget(bool doClip);
    void updateCurrentItem();
    bool isFrameLoadComplete() const { return m_frameLoadComplete; }

    struct FrameToNavigate;
    static void recursiveGatherFramesToNavigate(LocalFrame&, Vector<FrameToNavigate>&, HistoryItem& targetItem, HistoryItem* fromItem);
    Ref<LocalFrame> protectedFrame() const;

    const WeakRef<LocalFrame> m_frame;

    RefPtr<HistoryItem> m_currentItem;
    RefPtr<HistoryItem> m_previousItem;
    RefPtr<HistoryItem> m_provisionalItem;
    RefPtr<HistoryItem> m_policyItem;

    bool m_frameLoadComplete;

    bool m_defersLoading;
    RefPtr<HistoryItem> m_deferredItem;
    FrameLoadType m_deferredFrameLoadType;
};

} // namespace WebCore
