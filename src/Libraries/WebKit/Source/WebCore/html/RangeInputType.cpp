/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#include "RangeInputType.h"

#include "Decimal.h"
#include "DocumentInlines.h"
#include "ElementChildIteratorInlines.h"
#include "ElementRareData.h"
#include "EventNames.h"
#include "HTMLCollection.h"
#include "HTMLDataListElement.h"
#include "HTMLInputElement.h"
#include "HTMLOptionElement.h"
#include "HTMLParserIdioms.h"
#include "InputTypeNames.h"
#include "KeyboardEvent.h"
#include "MouseEvent.h"
#include "NodeName.h"
#include "PlatformMouseEvent.h"
#include "RenderSlider.h"
#include "ScopedEventQueue.h"
#include "ScriptDisallowedScope.h"
#include "ShadowRoot.h"
#include "SliderThumbElement.h"
#include "StepRange.h"
#include "UserAgentParts.h"
#include <limits>
#include <wtf/MathExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(TOUCH_EVENTS)
#include "Touch.h"
#include "TouchEvent.h"
#include "TouchList.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RangeInputType);

using namespace HTMLNames;

static const int rangeDefaultMinimum = 0;
static const int rangeDefaultMaximum = 100;
static const int rangeDefaultStep = 1;
static const int rangeDefaultStepBase = 0;
static const int rangeStepScaleFactor = 1;
static const StepRange::StepDescription rangeStepDescription { rangeDefaultStep, rangeDefaultStepBase, rangeStepScaleFactor };

static Decimal ensureMaximum(const Decimal& proposedValue, const Decimal& minimum)
{
    return proposedValue >= minimum ? proposedValue : minimum;
}

RangeInputType::RangeInputType(HTMLInputElement& element)
    : InputType(Type::Range, element)
{
    ASSERT(needsShadowSubtree());
}

const AtomString& RangeInputType::formControlType() const
{
    return InputTypeNames::range();
}

double RangeInputType::valueAsDouble() const
{
    ASSERT(element());
    return parseToDoubleForNumberType(element()->value());
}

ExceptionOr<void> RangeInputType::setValueAsDecimal(const Decimal& newValue, TextFieldEventBehavior eventBehavior) const
{
    ASSERT(element());
    element()->setValue(serialize(newValue), eventBehavior);
    return { };
}

bool RangeInputType::typeMismatchFor(const String& value) const
{
    return !value.isEmpty() && !std::isfinite(parseToDoubleForNumberType(value));
}

bool RangeInputType::supportsRequired() const
{
    return false;
}

StepRange RangeInputType::createStepRange(AnyStepHandling anyStepHandling) const
{
    ASSERT(element());
    const Decimal stepBase = findStepBase(rangeDefaultStepBase);
    const Decimal minimum = parseToNumber(element()->attributeWithoutSynchronization(minAttr), rangeDefaultMinimum);
    const Decimal maximum = ensureMaximum(parseToNumber(element()->attributeWithoutSynchronization(maxAttr), rangeDefaultMaximum), minimum);

    const Decimal step = StepRange::parseStep(anyStepHandling, rangeStepDescription, element()->attributeWithoutSynchronization(stepAttr));
    return StepRange(stepBase, RangeLimitations::Valid, minimum, maximum, step, rangeStepDescription);
}

// FIXME: Should this work for untrusted input?
void RangeInputType::handleMouseDownEvent(MouseEvent& event)
{
    ASSERT(element());

    if (!hasCreatedShadowSubtree())
        return;

    if (element()->isDisabledFormControl())
        return;

    RefPtr targetNode = dynamicDowncast<Node>(event.target());
    if (!targetNode)
        return;

    ASSERT(element()->shadowRoot());
    if (targetNode != element() && !targetNode->isDescendantOf(element()->protectedUserAgentShadowRoot().get()))
        return;
    Ref thumb = typedSliderThumbElement();
    if (targetNode == thumb.ptr())
        return;
    thumb->dragFrom(event.absoluteLocation());
}

#if ENABLE(TOUCH_EVENTS)
void RangeInputType::handleTouchEvent(TouchEvent& event)
{
    ASSERT(element());

    if (!hasCreatedShadowSubtree())
        return;

#if ENABLE(IOS_TOUCH_EVENTS)
    typedSliderThumbElement().handleTouchEvent(event);
#else

    if (element()->isDisabledFormControl())
        return;

    if (event.type() == eventNames().touchendEvent) {
        event.setDefaultHandled();
        return;
    }

    RefPtr<TouchList> touches = event.targetTouches();
    if (touches->length() == 1) {
        typedSliderThumbElement().setPositionFromPoint(touches->item(0)->absoluteLocation());
        event.setDefaultHandled();
    }
#endif // ENABLE(IOS_TOUCH_EVENTS)
}
#endif // ENABLE(TOUCH_EVENTS)

void RangeInputType::disabledStateChanged()
{
    if (!hasCreatedShadowSubtree())
        return;
    typedSliderThumbElement().hostDisabledStateChanged();
}

auto RangeInputType::handleKeydownEvent(KeyboardEvent& event) -> ShouldCallBaseEventHandler
{
    ASSERT(element());

    if (element()->isDisabledFormControl())
        return ShouldCallBaseEventHandler::Yes;

    const String& key = event.keyIdentifier();

    const Decimal current = parseToNumberOrNaN(element()->value());
    ASSERT(current.isFinite());

    auto stepRange = createStepRange(AnyStepHandling::Reject);

    // FIXME: We can't use stepUp() for the step value "any". So, we increase
    // or decrease the value by 1/100 of the value range. Is it reasonable?
    const Decimal step = equalLettersIgnoringASCIICase(element()->attributeWithoutSynchronization(stepAttr), "any"_s) ? (stepRange.maximum() - stepRange.minimum()) / 100 : stepRange.step();
    const Decimal bigStep = std::max((stepRange.maximum() - stepRange.minimum()) / 10, step);

    bool isVertical = false;
    if (auto* renderer = element()->renderer())
        isVertical = renderer->style().usedAppearance() == StyleAppearance::SliderVertical;

    Decimal newValue;
    if (key == "Up"_s)
        newValue = current + step;
    else if (key == "Down"_s)
        newValue = current - step;
    else if (key == "Left"_s)
        newValue = isVertical ? current + step : current - step;
    else if (key == "Right"_s)
        newValue = isVertical ? current - step : current + step;
    else if (key == "PageUp"_s)
        newValue = current + bigStep;
    else if (key == "PageDown"_s)
        newValue = current - bigStep;
    else if (key == "Home"_s)
        newValue = isVertical ? stepRange.maximum() : stepRange.minimum();
    else if (key == "End"_s)
        newValue = isVertical ? stepRange.minimum() : stepRange.maximum();
    else
        return ShouldCallBaseEventHandler::Yes; // Did not match any key binding.

    newValue = stepRange.clampValue(newValue);

    if (newValue != current) {
        EventQueueScope scope;
        setValueAsDecimal(newValue, DispatchInputAndChangeEvent);
    }

    event.setDefaultHandled();
    return ShouldCallBaseEventHandler::Yes;
}

void RangeInputType::createShadowSubtree()
{
    ASSERT(needsShadowSubtree());
    ASSERT(element());
    ASSERT(element()->userAgentShadowRoot());

    Ref document = element()->document();

    Ref shadowRoot = *element()->userAgentShadowRoot();
    ScriptDisallowedScope::EventAllowedScope eventAllowedScope { shadowRoot };

    Ref track = HTMLDivElement::create(document);
    Ref container = SliderContainerElement::create(document);
    shadowRoot->appendChild(ContainerNode::ChildChange::Source::Parser, container);
    container->appendChild(ContainerNode::ChildChange::Source::Parser, track);

    track->setUserAgentPart(UserAgentParts::webkitSliderRunnableTrack());
    track->appendChild(ContainerNode::ChildChange::Source::Parser, SliderThumbElement::create(document));
}

HTMLElement* RangeInputType::sliderTrackElement() const
{
    ASSERT(element());

    if (!hasCreatedShadowSubtree())
        return nullptr;

    RefPtr root = element()->userAgentShadowRoot();
    ASSERT(root);
    ASSERT(root->firstChild()); // container
    ASSERT(root->firstChild()->isHTMLElement());
    ASSERT(root->firstChild()->firstChild()); // track

    if (!root)
        return nullptr;
    
    RefPtr container = childrenOfType<SliderContainerElement>(*root).first();
    if (!container)
        return nullptr;

    return childrenOfType<HTMLElement>(*container).first();
}

SliderThumbElement& RangeInputType::typedSliderThumbElement() const
{
    ASSERT(hasCreatedShadowSubtree());
    ASSERT(sliderTrackElement()->firstChild()); // thumb
    ASSERT(sliderTrackElement()->firstChild()->isHTMLElement());

    return static_cast<SliderThumbElement&>(*sliderTrackElement()->firstChild());
}

HTMLElement* RangeInputType::sliderThumbElement() const
{
    return &typedSliderThumbElement();
}

RenderPtr<RenderElement> RangeInputType::createInputRenderer(RenderStyle&& style)
{
    ASSERT(element());
    return createRenderer<RenderSlider>(*element(), WTFMove(style));
}

Decimal RangeInputType::parseToNumber(const String& src, const Decimal& defaultValue) const
{
    return parseToDecimalForNumberType(src, defaultValue);
}

String RangeInputType::serialize(const Decimal& value) const
{
    if (!value.isFinite())
        return String();
    return serializeForNumberType(value);
}

// FIXME: Could share this with BaseClickableWithKeyInputType and BaseCheckableInputType if we had a common base class.
bool RangeInputType::accessKeyAction(bool sendMouseEvents)
{
    RefPtr element = this->element();
    return InputType::accessKeyAction(sendMouseEvents) || (element && element->dispatchSimulatedClick(0, sendMouseEvents ? SendMouseUpDownEvents : SendNoEvents));
}

void RangeInputType::attributeChanged(const QualifiedName& name)
{
    switch (name.nodeName()) {
    case AttributeNames::maxAttr:
    case AttributeNames::minAttr:
    case AttributeNames::valueAttr:
        // Sanitize the value.
        if (auto* element = this->element()) {
            if (element->hasDirtyValue())
                element->setValue(element->value());
        }
        if (hasCreatedShadowSubtree())
            typedSliderThumbElement().setPositionFromValue();
        break;
    default:
        break;
    }
    InputType::attributeChanged(name);
}

void RangeInputType::setValue(const String& value, bool valueChanged, TextFieldEventBehavior eventBehavior, TextControlSetValueSelection selection)
{
    InputType::setValue(value, valueChanged, eventBehavior, selection);

    if (!valueChanged)
        return;

    if (eventBehavior == DispatchNoEvent) {
        ASSERT(element());
        element()->setTextAsOfLastFormControlChangeEvent(value);
    }

    if (hasCreatedShadowSubtree())
        typedSliderThumbElement().setPositionFromValue();
}

String RangeInputType::fallbackValue() const
{
    return serializeForNumberType(createStepRange(AnyStepHandling::Reject).defaultValue());
}

String RangeInputType::sanitizeValue(const String& proposedValue) const
{
    StepRange stepRange(createStepRange(AnyStepHandling::Reject));
    const Decimal proposedNumericValue = parseToNumber(proposedValue, stepRange.defaultValue());
    return serializeForNumberType(stepRange.clampValue(proposedNumericValue));
}

bool RangeInputType::shouldRespectListAttribute()
{
    return element() && element()->document().settings().dataListElementEnabled();
}

void RangeInputType::dataListMayHaveChanged()
{
    m_tickMarkValuesDirty = true;
    RefPtr<HTMLElement> sliderTrackElement = this->sliderTrackElement();
    if (sliderTrackElement && sliderTrackElement->renderer())
        sliderTrackElement->renderer()->setNeedsLayout();
}

void RangeInputType::updateTickMarkValues()
{
    if (!m_tickMarkValuesDirty)
        return;
    m_tickMarkValues.clear();
    m_tickMarkValuesDirty = false;
    ASSERT(element());
    auto dataList = element()->dataList();
    if (!dataList)
        return;
    Ref<HTMLCollection> options = dataList->options();
    m_tickMarkValues.reserveCapacity(options->length());
    for (unsigned i = 0; i < options->length(); ++i) {
        RefPtr<Node> node = options->item(i);
        HTMLOptionElement& optionElement = downcast<HTMLOptionElement>(*node);
        String optionValue = optionElement.value();
        if (!element()->isValidValue(optionValue))
            continue;
        m_tickMarkValues.append(parseToNumber(optionValue, Decimal::nan()));
    }
    m_tickMarkValues.shrinkToFit();
    std::sort(m_tickMarkValues.begin(), m_tickMarkValues.end());
}

std::optional<Decimal> RangeInputType::findClosestTickMarkValue(const Decimal& value)
{
    updateTickMarkValues();
    if (!m_tickMarkValues.size())
        return std::nullopt;

    size_t left = 0;
    size_t right = m_tickMarkValues.size();
    size_t middle;
    while (true) {
        ASSERT(left <= right);
        middle = left + (right - left) / 2;
        if (!middle)
            break;
        if (middle == m_tickMarkValues.size() - 1 && m_tickMarkValues[middle] < value) {
            middle++;
            break;
        }
        if (m_tickMarkValues[middle - 1] <= value && m_tickMarkValues[middle] >= value)
            break;

        if (m_tickMarkValues[middle] < value)
            left = middle;
        else
            right = middle;
    }

    std::optional<Decimal> closestLeft = middle ? std::make_optional(m_tickMarkValues[middle - 1]) : std::nullopt;
    std::optional<Decimal> closestRight = middle != m_tickMarkValues.size() ? std::make_optional(m_tickMarkValues[middle]) : std::nullopt;

    if (!closestLeft)
        return closestRight;
    if (!closestRight)
        return closestLeft;

    if (*closestRight - value < value - *closestLeft)
        return closestRight;

    return closestLeft;
}

} // namespace WebCore
