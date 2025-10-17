/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
#include "UserAgentStyle.h"

#include "CSSCounterStyleRegistry.h"
#include "CSSCounterStyleRule.h"
#include "CSSKeyframesRule.h"
#include "CSSValuePool.h"
#include "Chrome.h"
#include "ChromeClient.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "ElementInlines.h"
#include "FullscreenManager.h"
#include "HTMLAnchorElement.h"
#include "HTMLAttachmentElement.h"
#include "HTMLBRElement.h"
#include "HTMLBodyElement.h"
#include "HTMLDataListElement.h"
#include "HTMLDivElement.h"
#include "HTMLEmbedElement.h"
#include "HTMLHeadElement.h"
#include "HTMLHtmlElement.h"
#include "HTMLInputElement.h"
#include "HTMLMediaElement.h"
#include "HTMLMeterElement.h"
#include "HTMLObjectElement.h"
#include "HTMLProgressElement.h"
#include "HTMLSpanElement.h"
#include "MathMLElement.h"
#include "MediaQueryEvaluator.h"
#include "Page.h"
#include "Quirks.h"
#include "RenderTheme.h"
#include "RuleSetBuilder.h"
#include "SVGElement.h"
#include "StyleResolver.h"
#include "StyleSheetContents.h"
#include "UserAgentStyleSheets.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/text/MakeString.h>

namespace WebCore {
namespace Style {

using namespace HTMLNames;

RuleSet* UserAgentStyle::defaultStyle;
RuleSet* UserAgentStyle::defaultQuirksStyle;
RuleSet* UserAgentStyle::defaultPrintStyle;
unsigned UserAgentStyle::defaultStyleVersion;

StyleSheetContents* UserAgentStyle::defaultStyleSheet;
StyleSheetContents* UserAgentStyle::quirksStyleSheet;
StyleSheetContents* UserAgentStyle::svgStyleSheet;
StyleSheetContents* UserAgentStyle::mathMLStyleSheet;
StyleSheetContents* UserAgentStyle::mediaQueryStyleSheet;
StyleSheetContents* UserAgentStyle::popoverStyleSheet;
StyleSheetContents* UserAgentStyle::horizontalFormControlsStyleSheet;
StyleSheetContents* UserAgentStyle::htmlSwitchControlStyleSheet;
StyleSheetContents* UserAgentStyle::counterStylesStyleSheet;
StyleSheetContents* UserAgentStyle::viewTransitionsStyleSheet;
#if ENABLE(FULLSCREEN_API)
StyleSheetContents* UserAgentStyle::fullscreenStyleSheet;
#endif
#if ENABLE(SERVICE_CONTROLS)
StyleSheetContents* UserAgentStyle::imageControlsStyleSheet;
#endif
#if ENABLE(ATTACHMENT_ELEMENT)
StyleSheetContents* UserAgentStyle::attachmentStyleSheet;
#endif

static const MQ::MediaQueryEvaluator& screenEval()
{
    static NeverDestroyed<const MQ::MediaQueryEvaluator> staticScreenEval(screenAtom());
    return staticScreenEval;
}

static const MQ::MediaQueryEvaluator& printEval()
{
    static NeverDestroyed<const MQ::MediaQueryEvaluator> staticPrintEval(printAtom());
    return staticPrintEval;
}

static StyleSheetContents* parseUASheet(const String& str)
{
    StyleSheetContents& sheet = StyleSheetContents::create(CSSParserContext(UASheetMode)).leakRef(); // leak the sheet on purpose
    sheet.parseString(str);
    return &sheet;
}
void static addToCounterStyleRegistry(StyleSheetContents& sheet)
{
    for (auto& rule : sheet.childRules()) {
        if (auto* counterStyleRule = dynamicDowncast<StyleRuleCounterStyle>(rule.get()))
            CSSCounterStyleRegistry::addUserAgentCounterStyle(counterStyleRule->descriptors());
    }
    CSSCounterStyleRegistry::resolveUserAgentReferences();
}

void static addUserAgentKeyframes(StyleSheetContents& sheet)
{
    // This does not handle nested rules.
    for (auto& rule : sheet.childRules()) {
        if (auto* styleRuleKeyframes = dynamicDowncast<StyleRuleKeyframes>(rule.get()))
            Style::Resolver::addUserAgentKeyframeStyle(*styleRuleKeyframes);
    }
}

void UserAgentStyle::addToDefaultStyle(StyleSheetContents& sheet)
{
    RuleSetBuilder screenBuilder(*defaultStyle, screenEval());
    screenBuilder.addRulesFromSheet(sheet);

    RuleSetBuilder printBuilder(*defaultPrintStyle, printEval());
    printBuilder.addRulesFromSheet(sheet);

    // Build a stylesheet consisting of non-trivial media queries seen in default style.
    // Rulesets for these can't be global and need to be built in document context.
    for (auto& rule : sheet.childRules()) {
        auto mediaRule = dynamicDowncast<StyleRuleMedia>(rule);
        if (!mediaRule)
            continue;
        auto& mediaQuery = mediaRule->mediaQueries();
        if (screenEval().evaluate(mediaQuery))
            continue;
        if (printEval().evaluate(mediaQuery))
            continue;
        mediaQueryStyleSheet->parserAppendRule(mediaRule->copy());
    }

    ++defaultStyleVersion;
}

void UserAgentStyle::initDefaultStyleSheet()
{
    if (defaultStyle)
        return;

    defaultStyle = &RuleSet::create().leakRef();
    defaultPrintStyle = &RuleSet::create().leakRef();
    defaultQuirksStyle = &RuleSet::create().leakRef();
    mediaQueryStyleSheet = &StyleSheetContents::create(CSSParserContext(UASheetMode)).leakRef();

    String defaultRules;
    auto extraDefaultStyleSheet = RenderTheme::singleton().extraDefaultStyleSheet();
    if (extraDefaultStyleSheet.isEmpty())
        defaultRules = StringImpl::createWithoutCopying(htmlUserAgentStyleSheet);
    else
        defaultRules = makeString(std::span { htmlUserAgentStyleSheet }, extraDefaultStyleSheet);
    defaultStyleSheet = parseUASheet(defaultRules);
    addToDefaultStyle(*defaultStyleSheet);

    counterStylesStyleSheet = parseUASheet(StringImpl::createWithoutCopying(counterStylesUserAgentStyleSheet));
    addToCounterStyleRegistry(*counterStylesStyleSheet);

    quirksStyleSheet = parseUASheet(StringImpl::createWithoutCopying(quirksUserAgentStyleSheet));

    RuleSetBuilder quirkBuilder(*defaultQuirksStyle, screenEval());
    quirkBuilder.addRulesFromSheet(*quirksStyleSheet);

    ++defaultStyleVersion;
}

void UserAgentStyle::ensureDefaultStyleSheetsForElement(const Element& element)
{
    if (is<HTMLElement>(element)) {
        if (RefPtr input = dynamicDowncast<HTMLInputElement>(element)) {
            if (!htmlSwitchControlStyleSheet && input->isSwitch()) {
                htmlSwitchControlStyleSheet = parseUASheet(StringImpl::createWithoutCopying(htmlSwitchControlUserAgentStyleSheet));
                addToDefaultStyle(*htmlSwitchControlStyleSheet);
            }
        }
#if ENABLE(ATTACHMENT_ELEMENT)
        else if (!attachmentStyleSheet && is<HTMLAttachmentElement>(element)) {
            attachmentStyleSheet = parseUASheet(RenderTheme::singleton().attachmentStyleSheet());
            addToDefaultStyle(*attachmentStyleSheet);
        }
#endif // ENABLE(ATTACHMENT_ELEMENT)

        if (!popoverStyleSheet && element.document().settings().popoverAttributeEnabled() && element.hasAttributeWithoutSynchronization(popoverAttr)) {
            popoverStyleSheet = parseUASheet(StringImpl::createWithoutCopying(popoverUserAgentStyleSheet));
            addToDefaultStyle(*popoverStyleSheet);
        }

        if ((is<HTMLFormControlElement>(element) || is<HTMLMeterElement>(element) || is<HTMLProgressElement>(element)) && !element.document().settings().verticalFormControlsEnabled()) {
            if (!horizontalFormControlsStyleSheet) {
                horizontalFormControlsStyleSheet = parseUASheet(StringImpl::createWithoutCopying(horizontalFormControlsUserAgentStyleSheet));
                addToDefaultStyle(*horizontalFormControlsStyleSheet);
            }
        }

    } else if (is<SVGElement>(element)) {
        if (!svgStyleSheet) {
            svgStyleSheet = parseUASheet(StringImpl::createWithoutCopying(svgUserAgentStyleSheet));
            addToDefaultStyle(*svgStyleSheet);
        }
    }
#if ENABLE(MATHML)
    else if (is<MathMLElement>(element)) {
        if (!mathMLStyleSheet) {
            mathMLStyleSheet = parseUASheet(StringImpl::createWithoutCopying(mathmlUserAgentStyleSheet));
            addToDefaultStyle(*mathMLStyleSheet);
        }
    }
#endif // ENABLE(MATHML)

#if ENABLE(FULLSCREEN_API)
    if (CheckedPtr fullscreenManager = element.document().fullscreenManagerIfExists(); !fullscreenStyleSheet && fullscreenManager && fullscreenManager->isFullscreen()) {
        fullscreenStyleSheet = parseUASheet(StringImpl::createWithoutCopying(fullscreenUserAgentStyleSheet));
        addToDefaultStyle(*fullscreenStyleSheet);
    }
#endif // ENABLE(FULLSCREEN_API)

    if (!viewTransitionsStyleSheet && element.document().settings().viewTransitionsEnabled()) {
        viewTransitionsStyleSheet = parseUASheet(StringImpl::createWithoutCopying(viewTransitionsUserAgentStyleSheet));
        addToDefaultStyle(*viewTransitionsStyleSheet);
        addUserAgentKeyframes(*viewTransitionsStyleSheet);
    }

    ASSERT(defaultStyle->features().idsInRules.isEmpty());
}

} // namespace Style
} // namespace WebCore
