/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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

#include "FloatPoint.h"
#include "SVGFitToViewBox.h"
#include "SVGGraphicsElement.h"
#include "SVGZoomAndPan.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct DOMMatrix2DInit;
class SMILTimeContainer;
class SVGAngle;
class SVGLength;
class SVGMatrix;
class SVGNumber;
class SVGRect;
class SVGTransform;
class SVGViewElement;
class SVGViewSpec;

class SVGSVGElement final : public SVGGraphicsElement, public SVGFitToViewBox, public SVGZoomAndPan {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGSVGElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGSVGElement);
public: // DOM
    float currentScale() const;
    void setCurrentScale(float);

    SVGPoint& currentTranslate() { return m_currentTranslate; }
    FloatPoint currentTranslateValue() const { return m_currentTranslate->value(); }

    bool useCurrentView() const { return m_useCurrentView; }
    SVGViewSpec& currentView();

    Ref<NodeList> getIntersectionList(SVGRect&, SVGElement* referenceElement);
    Ref<NodeList> getEnclosureList(SVGRect&, SVGElement* referenceElement);
    static bool checkIntersection(Ref<SVGElement>&&, SVGRect&);
    static bool checkEnclosure(Ref<SVGElement>&&, SVGRect&);

    void deselectAll();

    static Ref<SVGNumber> createSVGNumber();
    static Ref<SVGLength> createSVGLength();
    static Ref<SVGAngle> createSVGAngle();
    static Ref<SVGPoint> createSVGPoint();
    static Ref<SVGMatrix> createSVGMatrix();
    static Ref<SVGRect> createSVGRect();
    static Ref<SVGTransform> createSVGTransform();
    static Ref<SVGTransform> createSVGTransformFromMatrix(DOMMatrix2DInit&&);

    Element* getElementById(const AtomString&);

    void pauseAnimations();
    void unpauseAnimations();
    bool resumePausedAnimationsIfNeeded(const IntRect&);
    bool animationsPaused() const;
    bool hasActiveAnimation() const;
    float getCurrentTime() const;
    void setCurrentTime(float);
    
    unsigned suspendRedraw(unsigned) { return 1; }
    void unsuspendRedraw(unsigned) { }
    void unsuspendRedrawAll() { }
    void forceRedraw() { }

public:
    static Ref<SVGSVGElement> create(const QualifiedName&, Document&);
    static Ref<SVGSVGElement> create(Document&);
    bool scrollToFragment(StringView fragmentIdentifier);
    void resetScrollAnchor();

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGSVGElement, SVGGraphicsElement, SVGFitToViewBox>;
    using SVGGraphicsElement::ref;
    using SVGGraphicsElement::deref;

    SMILTimeContainer& timeContainer() { return m_timeContainer.get(); }
    Ref<SMILTimeContainer> protectedTimeContainer() const;

    void setCurrentTranslate(const FloatPoint&); // Used to pan.
    void updateCurrentTranslate();

    bool hasIntrinsicWidth() const;
    bool hasIntrinsicHeight() const;
    Length intrinsicWidth() const;
    Length intrinsicHeight() const;

    FloatSize currentViewportSizeExcludingZoom() const;
    FloatRect currentViewBoxRect() const;

    AffineTransform viewBoxToViewTransform(float viewWidth, float viewHeight) const;
    bool hasTransformRelatedAttributes() const final;

    const SVGLengthValue& x() const { return m_x->currentValue(); }
    const SVGLengthValue& y() const { return m_y->currentValue(); }
    const SVGLengthValue& width() const { return m_width->currentValue(); }
    const SVGLengthValue& height() const { return m_height->currentValue(); }

    SVGAnimatedLength& xAnimated() { return m_x; }
    SVGAnimatedLength& yAnimated() { return m_y; }
    SVGAnimatedLength& widthAnimated() { return m_width; }
    SVGAnimatedLength& heightAnimated() { return m_height; }

    void inheritViewAttributes(const SVGViewElement&);

private:
    SVGSVGElement(const QualifiedName&, Document&);
    virtual ~SVGSVGElement();

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;
    bool selfHasRelativeLengths() const override;
    bool isValid() const override;

    bool rendererIsNeeded(const RenderStyle&) override;
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    bool isReplaced(const RenderStyle&) const final;
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) override;
    void removedFromAncestor(RemovalType, ContainerNode&) override;
    void prepareForDocumentSuspension() override;
    void resumeFromDocumentSuspension() override;
    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) override;

    AffineTransform localCoordinateSpaceTransform(SVGLocatable::CTMScope) const override;
    RefPtr<LocalFrame> frameForCurrentScale() const;
    Ref<NodeList> collectIntersectionOrEnclosureList(SVGRect&, SVGElement*, bool (*checkFunction)(SVGElement&, SVGRect&));

    RefPtr<SVGViewElement> findViewAnchor(StringView fragmentIdentifier) const;
    SVGSVGElement* findRootAnchor(const SVGViewElement*) const;
    SVGSVGElement* findRootAnchor(StringView) const;

    bool m_useCurrentView { false };
    Ref<SMILTimeContainer> m_timeContainer;
    RefPtr<SVGViewSpec> m_viewSpec;
    RefPtr<SVGViewElement> m_currentViewElement;
    String m_currentViewFragmentIdentifier;

    Ref<SVGPoint> m_currentTranslate { SVGPoint::create() };

    Ref<SVGAnimatedLength> m_x { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_y { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedLength> m_width { SVGAnimatedLength::create(this, SVGLengthMode::Width, "100%"_s) };
    Ref<SVGAnimatedLength> m_height { SVGAnimatedLength::create(this, SVGLengthMode::Height, "100%"_s) };
};

} // namespace WebCore
