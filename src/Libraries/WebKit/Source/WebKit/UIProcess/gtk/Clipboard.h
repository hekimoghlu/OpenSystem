/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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

#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

#if USE(GTK4)
typedef struct _GdkClipboard GdkClipboard;
#else
typedef struct _GtkClipboard GtkClipboard;
#endif

namespace WebCore {
class SelectionData;
class SharedBuffer;
}

namespace WebKit {

class WebFrameProxy;

class Clipboard {
    WTF_MAKE_TZONE_ALLOCATED(Clipboard);
    WTF_MAKE_NONCOPYABLE(Clipboard);
public:
    static Clipboard& get(const String& name);

    enum class Type { Clipboard, Primary };
    explicit Clipboard(Type);
    ~Clipboard();

    enum class ReadMode : uint8_t { Asynchronous, Synchronous };

    Type type() const;
    void formats(CompletionHandler<void(Vector<String>&&)>&&);
    void readText(CompletionHandler<void(String&&)>&&, ReadMode = ReadMode::Asynchronous);
    void readFilePaths(CompletionHandler<void(Vector<String>&&)>&&, ReadMode = ReadMode::Asynchronous);
    void readURL(CompletionHandler<void(String&& url, String&& title)>&&, ReadMode = ReadMode::Asynchronous);
    void readBuffer(const char*, CompletionHandler<void(Ref<WebCore::SharedBuffer>&&)>&&, ReadMode = ReadMode::Asynchronous);
    void write(WebCore::SelectionData&&, CompletionHandler<void(int64_t)>&&);
    void clear();

    int64_t changeCount() const { return m_changeCount; }

private:
#if USE(GTK4)
    GdkClipboard* m_clipboard { nullptr };
#else
    GtkClipboard* m_clipboard { nullptr };
    WebFrameProxy* m_frameWritingToClipboard { nullptr };
#endif
    int64_t m_changeCount { 0 };
};

} // namespace WebKit
