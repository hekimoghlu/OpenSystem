/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#include "FontFaceSet.h"

#include "DOMPromiseProxy.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "EventLoop.h"
#include "FontFace.h"
#include "FrameDestructionObserverInlines.h"
#include "FrameLoader.h"
#include "JSDOMBinding.h"
#include "JSDOMPromiseDeferred.h"
#include "JSFontFace.h"
#include "JSFontFaceSet.h"
#include "Quirks.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FontFaceSet);

Ref<FontFaceSet> FontFaceSet::create(ScriptExecutionContext& context, const Vector<Ref<FontFace>>& initialFaces)
{
    Ref<FontFaceSet> result = adoptRef(*new FontFaceSet(context, initialFaces));
    result->suspendIfNeeded();
    return result;
}

Ref<FontFaceSet> FontFaceSet::create(ScriptExecutionContext& context, CSSFontFaceSet& backing)
{
    Ref<FontFaceSet> result = adoptRef(*new FontFaceSet(context, backing));
    result->suspendIfNeeded();
    return result;
}

FontFaceSet::FontFaceSet(ScriptExecutionContext& context, const Vector<Ref<FontFace>>& initialFaces)
    : ActiveDOMObject(&context)
    , m_backing(CSSFontFaceSet::create())
    , m_readyPromise(makeUniqueRef<ReadyPromise>(*this, &FontFaceSet::readyPromiseResolve))
{
    m_backing->addFontEventClient(*this);
    for (auto& face : initialFaces)
        add(face);
}

FontFaceSet::FontFaceSet(ScriptExecutionContext& context, CSSFontFaceSet& backing)
    : ActiveDOMObject(&context)
    , m_backing(backing)
    , m_readyPromise(makeUniqueRef<ReadyPromise>(*this, &FontFaceSet::readyPromiseResolve))
{
    if (auto* document = dynamicDowncast<Document>(context)) {
        if (document->frame())
            m_isDocumentLoaded = document->loadEventFinished() && !document->processingLoadEvent();
    }

    if (m_isDocumentLoaded && !backing.hasActiveFontFaces())
        m_readyPromise->resolve(*this);

    m_backing->addFontEventClient(*this);
}

FontFaceSet::~FontFaceSet() = default;

FontFaceSet::Iterator::Iterator(FontFaceSet& set)
    : m_target(set)
{
}

RefPtr<FontFace> FontFaceSet::Iterator::next()
{
    if (m_index >= m_target->size())
        return nullptr;
    return m_target->backing()[m_index++].wrapper(m_target->scriptExecutionContext());
}

FontFaceSet::PendingPromise::PendingPromise(LoadPromise&& promise)
    : promise(makeUniqueRef<LoadPromise>(WTFMove(promise)))
{
}

FontFaceSet::PendingPromise::~PendingPromise() = default;

bool FontFaceSet::has(FontFace& face) const
{
    if (face.backing().cssConnection())
        m_backing->updateStyleIfNeeded();
    return m_backing->hasFace(face.backing());
}

size_t FontFaceSet::size()
{
    m_backing->updateStyleIfNeeded();
    return m_backing->faceCount();
}

ExceptionOr<FontFaceSet&> FontFaceSet::add(FontFace& face)
{
    if (m_backing->hasFace(face.backing()))
        return *this;
    if (face.backing().cssConnection())
        return Exception(ExceptionCode::InvalidModificationError);
    m_backing->add(face.backing());
    return *this;
}

bool FontFaceSet::remove(FontFace& face)
{
    if (face.backing().cssConnection())
        return false;
    bool result = m_backing->hasFace(face.backing());
    if (result)
        m_backing->remove(face.backing());
    return result;
}

void FontFaceSet::clear()
{
    auto facesPartitionIndex = m_backing->facesPartitionIndex();
    while (m_backing->faceCount() > facesPartitionIndex) {
        m_backing->remove(m_backing.get()[m_backing->faceCount() - 1]);
        ASSERT(m_backing->facesPartitionIndex() == facesPartitionIndex);
    }
}

void FontFaceSet::load(const String& font, const String& text, LoadPromise&& promise)
{
    m_backing->updateStyleIfNeeded();
    auto matchingFacesResult = m_backing->matchingFacesExcludingPreinstalledFonts(font, text);
    if (matchingFacesResult.hasException()) {
        promise.reject(matchingFacesResult.releaseException());
        return;
    }
    auto matchingFaces = matchingFacesResult.releaseReturnValue();

    if (matchingFaces.isEmpty()) {
        promise.resolve({ });
        return;
    }

    for (auto& face : matchingFaces)
        face.get().load();

    auto* document = dynamicDowncast<Document>(scriptExecutionContext());
    if (document && document->quirks().shouldEnableFontLoadingAPIQuirk()) {
        // HBOMax.com expects that loading fonts will succeed, and will totally break when it doesn't. But when lockdown mode is enabled, fonts
        // fail to load, because that's the whole point of lockdown mode.
        //
        // This is a bit of a hack to say "When lockdown mode is enabled, and lockdown mode has removed all the remote fonts, then just pretend
        // that the fonts loaded successfully." If there are any non-remote fonts still present, don't make any behavior change.
        //
        // See also: https://github.com/w3c/csswg-drafts/issues/7680

        bool hasSource = false;
        for (auto& face : matchingFaces) {
            if (face.get().sourceCount()) {
                hasSource = true;
                break;
            }
        }
        if (!hasSource) {
            promise.resolve(matchingFaces.map([scriptExecutionContext = scriptExecutionContext()] (const auto& matchingFace) {
                return matchingFace.get().wrapper(scriptExecutionContext);
            }));
            return;
        }
    }

    for (auto& face : matchingFaces) {
        if (face.get().status() == CSSFontFace::Status::Failure) {
            promise.reject(ExceptionCode::NetworkError);
            return;
        }
    }

    auto pendingPromise = PendingPromise::create(WTFMove(promise));
    bool waiting = false;

    for (auto& face : matchingFaces) {
        pendingPromise->faces.append(face.get().wrapper(scriptExecutionContext()));
        if (face.get().status() == CSSFontFace::Status::Success)
            continue;
        waiting = true;
        ASSERT(face.get().existingWrapper());
        m_pendingPromises.add(face.get().existingWrapper(), Vector<Ref<PendingPromise>>()).iterator->value.append(pendingPromise.copyRef());
    }

    if (!waiting)
        pendingPromise->promise->resolve(pendingPromise->faces);
}

ExceptionOr<bool> FontFaceSet::check(const String& family, const String& text)
{
    m_backing->updateStyleIfNeeded();
    return m_backing->check(family, text);
}
    
auto FontFaceSet::status() const -> LoadStatus
{
    m_backing->updateStyleIfNeeded();

    switch (m_backing->status()) {
    case CSSFontFaceSet::Status::Loading:
        return LoadStatus::Loading;
    case CSSFontFaceSet::Status::Loaded:
        return LoadStatus::Loaded;
    }
    ASSERT_NOT_REACHED();
    return LoadStatus::Loaded;
}

void FontFaceSet::faceFinished(CSSFontFace& face, CSSFontFace::Status newStatus)
{
    if (!face.existingWrapper())
        return;

    auto pendingPromises = m_pendingPromises.take(face.existingWrapper());
    if (pendingPromises.isEmpty())
        return;

    for (auto& pendingPromise : pendingPromises) {
        if (pendingPromise->hasReachedTerminalState)
            continue;
        if (newStatus == CSSFontFace::Status::Success) {
            if (pendingPromise->hasOneRef()) {
                pendingPromise->promise->resolve(pendingPromise->faces);
                pendingPromise->hasReachedTerminalState = true;
            }
        } else {
            ASSERT(newStatus == CSSFontFace::Status::Failure);
            pendingPromise->promise->reject(ExceptionCode::NetworkError);
            pendingPromise->hasReachedTerminalState = true;
        }
    }
}

void FontFaceSet::startedLoading()
{
    // FIXME: Fire a "loading" event asynchronously.
}

void FontFaceSet::documentDidFinishLoading()
{
    m_isDocumentLoaded = true;
    if (!m_backing->hasActiveFontFaces() && !m_readyPromise->isFulfilled())
        m_readyPromise->resolve(*this);
}

void FontFaceSet::completedLoading()
{
    if (m_isDocumentLoaded && !m_readyPromise->isFulfilled())
        m_readyPromise->resolve(*this);
}

FontFaceSet& FontFaceSet::readyPromiseResolve()
{
    return *this;
}

}
