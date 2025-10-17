/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#include "ColorInputType.h"

#include "AXObjectCache.h"
#include "CSSPropertyParserConsumer+Color.h"
#include "Chrome.h"
#include "Color.h"
#include "ColorSerialization.h"
#include "ColorTypes.h"
#include "ElementChildIteratorInlines.h"
#include "ElementRareData.h"
#include "Event.h"
#include "HTMLDataListElement.h"
#include "HTMLDivElement.h"
#include "HTMLInputElement.h"
#include "HTMLOptionElement.h"
#include "InputTypeNames.h"
#include "RenderTheme.h"
#include "RenderView.h"
#include "ScopedEventQueue.h"
#include "ScriptDisallowedScope.h"
#include "ShadowRoot.h"
#include "TypedElementDescendantIteratorInlines.h"
#include "UserAgentParts.h"
#include "UserGestureIndicator.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ColorInputType);

using namespace HTMLNames;

using LazySlowPathColorParsingParameters = std::tuple<
    CSSPropertyParserHelpers::CSSColorParsingOptions,
    CSS::PlatformColorResolutionState,
    std::optional<CSS::PlatformColorResolutionDelegate>
>;

// https://html.spec.whatwg.org/multipage/infrastructure.html#valid-simple-colour
static bool isValidSimpleColor(StringView string)
{
    if (string.length() != 7)
        return false;
    if (string[0] != '#')
        return false;
    for (unsigned i = 1; i < 7; ++i) {
        if (!isASCIIHexDigit(string[i]))
            return false;
    }
    return true;
}

// https://html.spec.whatwg.org/multipage/common-microsyntaxes.html#rules-for-parsing-simple-colour-values
static std::optional<SRGBA<uint8_t>> parseSimpleColorValue(StringView string)
{
    if (!isValidSimpleColor(string))
        return std::nullopt;
    return { { toASCIIHexValue(string[1], string[2]), toASCIIHexValue(string[3], string[4]), toASCIIHexValue(string[5], string[6]) } };
}

static LazySlowPathColorParsingParameters colorParsingParameters()
{
    return {
        CSSPropertyParserHelpers::CSSColorParsingOptions {
            .allowedColorTypes = { CSS::ColorType::Absolute, CSS::ColorType::Current, CSS::ColorType::System }
        },
        CSS::PlatformColorResolutionState {
            .resolvedCurrentColor = Color::black
        },
        std::nullopt
    };
}

static std::optional<Color> parseColorValue(StringView string, HTMLInputElement& context)
{
    if (context.colorSpace().isNull())
        return parseSimpleColorValue(string);

    auto parserContext = context.document().cssParserContext();
    parserContext.mode = HTMLStandardMode;
    auto color = CSSPropertyParserHelpers::parseColorRaw(string.toString(), parserContext, [] {
        return colorParsingParameters();
    });

    if (!color.isValid())
        return { };

    return color;
}

static String serializeColorValue(Color input, HTMLInputElement& context)
{
    auto alpha = context.alpha();
    auto colorSpace = context.colorSpace();

    if (!alpha)
        input = input.opaqueColor();

    if (colorSpace.isNull() || colorSpace == "limited-srgb"_s) {
        auto inputAsRGBA = input.toColorTypeLossy<SRGBA<uint8_t>>();
        // When the alpha attribute is set the specification requires the modern color() serialization.
        if (alpha)
            input = { inputAsRGBA, { Color::Flags::UseColorFunctionSerialization } };
        else
            input = inputAsRGBA.resolved();
    } else {
        ASSERT(colorSpace == "display-p3"_s);
        input = input.toColorTypeLossy<ExtendedDisplayP3<float>>().resolved();
    }

    return serializationForHTML(input);
}

ColorInputType::~ColorInputType()
{
    endColorChooser();
}

bool ColorInputType::isMouseFocusable() const
{
    ASSERT(element());
    return element()->isTextFormControlFocusable();
}

bool ColorInputType::isKeyboardFocusable(KeyboardEvent*) const
{
    ASSERT(element());
#if PLATFORM(IOS_FAMILY)
    return element()->isTextFormControlFocusable();
#else
    return false;
#endif
}

bool ColorInputType::isPresentingAttachedView() const
{
    return !!m_chooser;
}

const AtomString& ColorInputType::formControlType() const
{
    return InputTypeNames::color();
}

bool ColorInputType::supportsRequired() const
{
    return false;
}

String ColorInputType::fallbackValue() const
{
    ASSERT(element());
    return serializeColorValue(Color::black, *element());
}

String ColorInputType::sanitizeValue(const String& proposedValue) const
{
    ASSERT(element());
    auto color = parseColorValue(proposedValue, *element());

    if (!color)
        return fallbackValue();

    return serializeColorValue(*color, *element());
}

Color ColorInputType::valueAsColor() const
{
    ASSERT(element());
    auto color = parseColorValue(element()->value(), *element());
    ASSERT(!!color);
    return *color;
}

void ColorInputType::createShadowSubtree()
{
    ASSERT(needsShadowSubtree());
    ASSERT(element());
    ASSERT(element()->shadowRoot());

    Ref document = element()->document();
    Ref wrapperElement = HTMLDivElement::create(document);
    Ref colorSwatch = HTMLDivElement::create(document);

    Ref shadowRoot = *element()->userAgentShadowRoot();
    ScriptDisallowedScope::EventAllowedScope eventAllowedScope { shadowRoot };
    shadowRoot->appendChild(ContainerNode::ChildChange::Source::Parser, wrapperElement);

    wrapperElement->appendChild(ContainerNode::ChildChange::Source::Parser, colorSwatch);
    wrapperElement->setUserAgentPart(UserAgentParts::webkitColorSwatchWrapper());
    colorSwatch->setUserAgentPart(UserAgentParts::webkitColorSwatch());

    RenderTheme::singleton().createColorWellSwatchSubtree(colorSwatch.get());

    updateColorSwatch();
}

void ColorInputType::setValue(const String& value, bool valueChanged, TextFieldEventBehavior eventBehavior, TextControlSetValueSelection selection)
{
    InputType::setValue(value, valueChanged, eventBehavior, selection);

    if (!valueChanged)
        return;

    updateColorSwatch();
    if (RefPtr chooser = m_chooser)
        chooser->setSelectedColor(valueAsColor());
}

void ColorInputType::attributeChanged(const QualifiedName& name)
{
    if (name == valueAttr) {
        updateColorSwatch();

        if (CheckedPtr cache = element()->document().existingAXObjectCache())
            cache->valueChanged(*element());
    }

    InputType::attributeChanged(name);
}

void ColorInputType::handleDOMActivateEvent(Event& event)
{
    ASSERT(element());
    if (element()->isDisabledFormControl() || !element()->renderer())
        return;

    if (!UserGestureIndicator::processingUserGesture())
        return;

    showPicker();
    event.setDefaultHandled();
}

void ColorInputType::showPicker() 
{
    if (Chrome* chrome = this->chrome()) {
        if (RefPtr chooser = m_chooser)
            chooser->reattachColorChooser(valueAsColor());
        else
            m_chooser = chrome->createColorChooser(*this, valueAsColor());
    }
}

bool ColorInputType::allowsShowPickerAcrossFrames()
{
    return true;
}

void ColorInputType::detach()
{
    endColorChooser();
}

void ColorInputType::elementDidBlur()
{
    endColorChooser();
}

bool ColorInputType::shouldRespectListAttribute()
{
    return true;
}

bool ColorInputType::shouldResetOnDocumentActivation()
{
    return true;
}

void ColorInputType::didChooseColor(const Color& color)
{
    ASSERT(element());

    if (element()->isDisabledFormControl())
        return;

    auto serializedColor = serializeColorValue(color, *element());
    if (serializedColor == element()->value())
        return;

    EventQueueScope scope;
    element()->setValueFromRenderer(serializedColor);
    updateColorSwatch();
    element()->dispatchFormControlChangeEvent();

    if (CheckedPtr cache = element()->document().existingAXObjectCache())
        cache->valueChanged(*element());
}

void ColorInputType::didEndChooser()
{
    m_chooser = nullptr;
    if (element()->renderer())
        element()->renderer()->repaint();
}

void ColorInputType::endColorChooser()
{
    if (RefPtr chooser = m_chooser)
        chooser->endChooser();
}

void ColorInputType::updateColorSwatch()
{
    RefPtr colorSwatch = shadowColorSwatch();
    if (!colorSwatch)
        return;

    RenderTheme::singleton().setColorWellSwatchBackground(*colorSwatch, valueAsColor());
}

HTMLElement* ColorInputType::shadowColorSwatch() const
{
    ASSERT(element());
    RefPtr shadow = element()->userAgentShadowRoot();
    if (!shadow)
        return nullptr;

    RefPtr wrapper = childrenOfType<HTMLDivElement>(*shadow).first();
    if (!wrapper)
        return nullptr;

    return childrenOfType<HTMLDivElement>(*wrapper).first();
}

IntRect ColorInputType::elementRectRelativeToRootView() const
{
    ASSERT(element());
    if (!element()->renderer())
        return IntRect();
    return element()->document().view()->contentsToRootView(element()->renderer()->absoluteBoundingBoxRect());
}

bool ColorInputType::supportsAlpha() const
{
    ASSERT(element());
    return element()->alpha();
}

Vector<Color> ColorInputType::suggestedColors() const
{
    Vector<Color> suggestions;
    ASSERT(element());
    if (auto dataList = element()->dataList()) {
        for (auto& option : dataList->suggestions()) {
            if (auto color = parseColorValue(option.value(), *element()))
                suggestions.append(*color);
        }
    }
    return suggestions;
}

void ColorInputType::selectColor(StringView string)
{
    ASSERT(element());
    if (auto color = parseColorValue(string, *element()))
        didChooseColor(*color);
}

} // namespace WebCore
