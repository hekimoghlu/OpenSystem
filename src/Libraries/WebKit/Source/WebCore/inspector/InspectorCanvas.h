/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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

#include "InspectorCanvasCallTracer.h"
#include <JavaScriptCore/AsyncStackTrace.h>
#include <JavaScriptCore/InspectorProtocolObjects.h>
#include <JavaScriptCore/ScriptCallFrame.h>
#include <JavaScriptCore/ScriptCallStack.h>
#include <variant>
#include <wtf/HashSet.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class CSSStyleImageValue;
class CanvasGradient;
class CanvasPattern;
class Element;
class HTMLCanvasElement;
class HTMLImageElement;
class HTMLVideoElement;
class ImageBitmap;
class ImageData;
class OffscreenCanvas;

class InspectorCanvas final : public RefCounted<InspectorCanvas> {
public:
    static Ref<InspectorCanvas> create(CanvasRenderingContext&);

    const String& identifier() const { return m_identifier; }

    const CanvasRenderingContext& canvasContext() const { return m_context.get(); }
    CanvasRenderingContext& canvasContext() { return m_context.get(); }
    HTMLCanvasElement* canvasElement() const;

    ScriptExecutionContext* scriptExecutionContext() const;

    JSC::JSValue resolveContext(JSC::JSGlobalObject*);

    UncheckedKeyHashSet<Element*> clientNodes() const;

    void canvasChanged();

    void resetRecordingData();
    bool hasRecordingData() const;
    bool currentFrameHasData() const;

    // InspectorCanvasCallTracer
#define PROCESS_ARGUMENT_DECLARATION(ArgumentType) \
    std::optional<InspectorCanvasCallTracer::ProcessedArgument> processArgument(ArgumentType); \
// end of PROCESS_ARGUMENT_DECLARATION
    FOR_EACH_INSPECTOR_CANVAS_CALL_TRACER_ARGUMENT(PROCESS_ARGUMENT_DECLARATION)
#undef PROCESS_ARGUMENT_DECLARATION
    void recordAction(String&&, InspectorCanvasCallTracer::ProcessedArguments&& = { });

    Ref<JSON::ArrayOf<Inspector::Protocol::Recording::Frame>> releaseFrames() { return m_frames.releaseNonNull(); }

    void finalizeFrame();
    void markCurrentFrameIncomplete();

    void setRecordingName(const String& name) { m_recordingName = name; }

    void setBufferLimit(long);
    bool hasBufferSpace() const;
    long bufferUsed() const { return m_bufferUsed; }

    void setFrameCount(long);
    bool overFrameCount() const;

    Ref<Inspector::Protocol::Canvas::Canvas> buildObjectForCanvas(bool captureBacktrace);
    Ref<Inspector::Protocol::Recording::Recording> releaseObjectForRecording();

    static Inspector::Protocol::ErrorStringOr<String> getContentAsDataURL(CanvasRenderingContext&);
    Inspector::Protocol::ErrorStringOr<String> getContentAsDataURL() { return getContentAsDataURL(m_context); };

private:
    explicit InspectorCanvas(CanvasRenderingContext&);

    void appendActionSnapshotIfNeeded();

    using DuplicateDataVariant = std::variant<
        RefPtr<CanvasGradient>,
        RefPtr<CanvasPattern>,
        RefPtr<HTMLCanvasElement>,
        RefPtr<HTMLImageElement>,
#if ENABLE(VIDEO)
        RefPtr<HTMLVideoElement>,
#endif
        RefPtr<ImageData>,
        RefPtr<ImageBitmap>,
        RefPtr<Inspector::ScriptCallStack>,
        RefPtr<Inspector::AsyncStackTrace>,
        RefPtr<CSSStyleImageValue>,
        Inspector::ScriptCallFrame,
#if ENABLE(OFFSCREEN_CANVAS)
        RefPtr<OffscreenCanvas>,
#endif
        String
    >;

    int indexForData(DuplicateDataVariant);
    Ref<JSON::Value> valueIndexForData(DuplicateDataVariant);
    String stringIndexForKey(const String&);
    Ref<Inspector::Protocol::Recording::InitialState> buildInitialState();
    Ref<JSON::ArrayOf<JSON::Value>> buildAction(String&&, InspectorCanvasCallTracer::ProcessedArguments&& = { });
    Ref<JSON::ArrayOf<JSON::Value>> buildArrayForCanvasGradient(const CanvasGradient&);
    Ref<JSON::ArrayOf<JSON::Value>> buildArrayForCanvasPattern(const CanvasPattern&);
    Ref<JSON::ArrayOf<JSON::Value>> buildArrayForImageData(const ImageData&);

    String m_identifier;

    WeakRef<CanvasRenderingContext> m_context;

    RefPtr<Inspector::Protocol::Recording::InitialState> m_initialState;
    RefPtr<JSON::ArrayOf<Inspector::Protocol::Recording::Frame>> m_frames;
    RefPtr<JSON::ArrayOf<JSON::Value>> m_currentActions;
    RefPtr<JSON::ArrayOf<JSON::Value>> m_lastRecordedAction;
    RefPtr<JSON::ArrayOf<JSON::Value>> m_serializedDuplicateData;
    Vector<DuplicateDataVariant> m_indexedDuplicateData;

    String m_recordingName;
    MonotonicTime m_currentFrameStartTime { MonotonicTime::nan() };
    size_t m_bufferLimit { 100 * 1024 * 1024 };
    size_t m_bufferUsed { 0 };
    std::optional<size_t> m_frameCount;
    size_t m_framesCaptured { 0 };
    bool m_contentChanged { false };
};

} // namespace WebCore
