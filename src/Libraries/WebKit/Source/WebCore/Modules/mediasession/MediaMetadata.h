/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

#if ENABLE(MEDIA_SESSION)

#include "CachedImageClient.h"
#include "CachedResourceHandle.h"
#include "MediaMetadataInit.h"
#include "MediaSession.h"
#include <wtf/CheckedRef.h>
#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class CachedImage;
class Document;
class Image;
class WeakPtrImplWithEventTargetData;
struct MediaImage;

using MediaSessionMetadata = MediaMetadataInit;

class ArtworkImageLoader final : public CachedImageClient {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ArtworkImageLoader, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ArtworkImageLoader);
public:
    using ArtworkImageLoaderCallback = Function<void(Image*)>;
    // The callback will only be called upon success or explicit failure to retrieve the image. If the operation is interrupted following the
    // destruction of the ArtworkImageLoader, the callback won't be called.
    WEBCORE_EXPORT ArtworkImageLoader(Document&, const String& src, ArtworkImageLoaderCallback&&);
    WEBCORE_EXPORT ~ArtworkImageLoader();

    WEBCORE_EXPORT void requestImageResource();

protected:
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) override;

private:
    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
    const String m_src;
    ArtworkImageLoaderCallback m_callback;
    CachedResourceHandle<CachedImage> m_cachedImage;
};

class MediaMetadata final : public RefCounted<MediaMetadata> {
public:
    static ExceptionOr<Ref<MediaMetadata>> create(ScriptExecutionContext&, std::optional<MediaMetadataInit>&&);
    static Ref<MediaMetadata> create(MediaSession&, Vector<URL>&&);
    ~MediaMetadata();

    void setMediaSession(MediaSession&);
    void resetMediaSession();

    const String& title() const { return m_metadata.title; }
    void setTitle(const String&);

    const String& artist() const { return m_metadata.artist; }
    void setArtist(const String&);

    const String& album() const { return m_metadata.album; }
    void setAlbum(const String&);

    const Vector<MediaImage>& artwork() const { return m_metadata.artwork; }
    ExceptionOr<void> setArtwork(ScriptExecutionContext&, Vector<MediaImage>&&);

    const String& artworkSrc() const { return m_artworkImageSrc; }
    const RefPtr<Image>& artworkImage() const { return m_artworkImage; }

    const MediaSessionMetadata& metadata() const { return m_metadata; }

#if ENABLE(MEDIA_SESSION_PLAYLIST)
    const String& trackIdentifier() const { return m_metadata.trackIdentifier; }
    void setTrackIdentifier(const String&);
#endif

private:
    struct Pair {
        float score;
        String src;
    };

    MediaMetadata();
    void setArtworkImage(Image*);
    void metadataUpdated();
    void refreshArtworkImage();
    void tryNextArtworkImage(uint32_t, Vector<Pair>&&);

    static constexpr int s_minimumSize = 128;
    static constexpr int s_idealSize = 512;

    WeakPtr<MediaSession> m_session;
    MediaSessionMetadata m_metadata;
    std::unique_ptr<ArtworkImageLoader> m_artworkLoader;
    String m_artworkImageSrc;
    RefPtr<Image> m_artworkImage;
    Vector<URL> m_defaultImages;
};

}

#endif
