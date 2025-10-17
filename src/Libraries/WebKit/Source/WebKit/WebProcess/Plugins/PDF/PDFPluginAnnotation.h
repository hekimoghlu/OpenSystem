/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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

#if ENABLE(PDF_PLUGIN)

#include <WebCore/EventListener.h>
#include <wtf/CheckedPtr.h>
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class Document;
class Element;
class WeakPtrImplWithEventTargetData;
}

OBJC_CLASS PDFAnnotation;
OBJC_CLASS PDFLayerController;

namespace WebKit {

class PDFPluginBase;

class PDFPluginAnnotation : public RefCounted<PDFPluginAnnotation>, public CanMakeCheckedPtr<PDFPluginAnnotation> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PDFPluginAnnotation);
public:
    static RefPtr<PDFPluginAnnotation> create(PDFAnnotation *, PDFPluginBase*);
    virtual ~PDFPluginAnnotation();

    WebCore::Element* element() const { return m_element.get(); }
    PDFAnnotation *annotation() const { return m_annotation.get(); }
    PDFPluginBase* plugin() const { return m_plugin.get(); }

    RefPtr<WebCore::Element> protectedElement() const { return element(); }

    virtual void updateGeometry();
    virtual void commit();

    void attach(WebCore::Element*);

protected:
    PDFPluginAnnotation(PDFAnnotation *annotation, PDFPluginBase* plugin)
        : m_annotation(annotation)
        , m_eventListener(PDFPluginAnnotationEventListener::create(this))
        , m_plugin(plugin)
    {
    }

    WebCore::Element* parent() const { return m_parent.get(); }
    WebCore::EventListener* eventListener() const { return m_eventListener.get(); }

    virtual bool handleEvent(WebCore::Event&);

private:
    virtual Ref<WebCore::Element> createAnnotationElement() = 0;

    class PDFPluginAnnotationEventListener : public WebCore::EventListener {
    public:
        static Ref<PDFPluginAnnotationEventListener> create(PDFPluginAnnotation* annotation)
        {
            return adoptRef(*new PDFPluginAnnotationEventListener(annotation));
        }

        void setAnnotation(PDFPluginAnnotation* annotation) { m_annotation = annotation; }

    private:

        PDFPluginAnnotationEventListener(PDFPluginAnnotation* annotation)
            : WebCore::EventListener(WebCore::EventListener::CPPEventListenerType)
            , m_annotation(annotation)
        {
        }

        void handleEvent(WebCore::ScriptExecutionContext&, WebCore::Event&) override;

        CheckedPtr<PDFPluginAnnotation> m_annotation;
    };

    WeakPtr<WebCore::Element, WebCore::WeakPtrImplWithEventTargetData> m_parent;

    RefPtr<WebCore::Element> m_element;
    RetainPtr<PDFAnnotation> m_annotation;

    RefPtr<PDFPluginAnnotationEventListener> m_eventListener;

    RefPtr<PDFPluginBase> m_plugin;
};

} // namespace WebKit

#endif // ENABLE(PDF_PLUGIN)
