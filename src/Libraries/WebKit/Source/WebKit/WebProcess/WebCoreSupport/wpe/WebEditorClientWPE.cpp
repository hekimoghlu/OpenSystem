/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#include "WebEditorClient.h"

#include "WebPage.h"
#include <WebCore/Document.h>
#include <WebCore/Editor.h>
#include <WebCore/EventNames.h>
#include <WebCore/FrameDestructionObserverInlines.h>
#include <WebCore/KeyboardEvent.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/Node.h>
#include <WebCore/PlatformKeyboardEvent.h>
#include <WebCore/WindowsKeyboardCodes.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {
using namespace WebCore;

// The idea for the array/map below comes from Blink's EditingBehavior.cpp.

static const unsigned CtrlKey  = 1 << 0;
static const unsigned AltKey   = 1 << 1;
static const unsigned ShiftKey = 1 << 2;
static const unsigned MetaKey  = 1 << 3;

// Keys with special meaning. These will be delegated to the editor using
// the execCommand() method
struct KeyDownEntry {
    unsigned virtualKey;
    unsigned modifiers;
    const char* name;
};

struct KeyPressEntry {
    unsigned charCode;
    unsigned modifiers;
    const char* name;
};

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // WPE port

static const KeyDownEntry keyDownEntries[] = {
    { VK_LEFT,   0,                  "MoveLeft"                                },
    { VK_LEFT,   ShiftKey,           "MoveLeftAndModifySelection"              },
    { VK_LEFT,   CtrlKey,            "MoveWordLeft"                            },
    { VK_LEFT,   CtrlKey | ShiftKey,
        "MoveWordLeftAndModifySelection"                                       },
    { VK_RIGHT,  0,                  "MoveRight"                               },
    { VK_RIGHT,  ShiftKey,           "MoveRightAndModifySelection"             },
    { VK_RIGHT,  CtrlKey,            "MoveWordRight"                           },
    { VK_RIGHT,  CtrlKey | ShiftKey, "MoveWordRightAndModifySelection"         },
    { VK_UP,     0,                  "MoveUp"                                  },
    { VK_UP,     ShiftKey,           "MoveUpAndModifySelection"                },
    { VK_PRIOR,  ShiftKey,           "MovePageUpAndModifySelection"            },
    { VK_DOWN,   0,                  "MoveDown"                                },
    { VK_DOWN,   ShiftKey,           "MoveDownAndModifySelection"              },
    { VK_NEXT,   ShiftKey,           "MovePageDownAndModifySelection"          },
    { VK_UP,     CtrlKey | ShiftKey, "MoveParagraphBackwardAndModifySelection" },
    { VK_DOWN,   CtrlKey | ShiftKey, "MoveParagraphForwardAndModifySelection"  },
    { VK_PRIOR,  0,                  "MovePageUp"                              },
    { VK_NEXT,   0,                  "MovePageDown"                            },
    { VK_HOME,   0,                  "MoveToBeginningOfLine"                   },
    { VK_HOME,   ShiftKey,
        "MoveToBeginningOfLineAndModifySelection"                              },
    { VK_HOME,   CtrlKey,            "MoveToBeginningOfDocument"               },
    { VK_HOME,   CtrlKey | ShiftKey,
        "MoveToBeginningOfDocumentAndModifySelection"                          },
    { VK_END,    0,                  "MoveToEndOfLine"                         },
    { VK_END,    ShiftKey,           "MoveToEndOfLineAndModifySelection"       },
    { VK_END,    CtrlKey,            "MoveToEndOfDocument"                     },
    { VK_END,    CtrlKey | ShiftKey,
        "MoveToEndOfDocumentAndModifySelection"                                },
    { VK_BACK,   0,                  "DeleteBackward"                          },
    { VK_BACK,   ShiftKey,           "DeleteBackward"                          },
    { VK_DELETE, 0,                  "DeleteForward"                           },
    { VK_BACK,   CtrlKey,            "DeleteWordBackward"                      },
    { VK_DELETE, CtrlKey,            "DeleteWordForward"                       },
    { 'B',         CtrlKey,          "ToggleBold"                              },
    { 'I',         CtrlKey,          "ToggleItalic"                            },
    { 'U',         CtrlKey,          "ToggleUnderline"                         },
    { VK_ESCAPE, 0,                  "Cancel"                                  },
    { VK_OEM_PERIOD, CtrlKey,        "Cancel"                                  },
    { VK_TAB,    0,                  "InsertTab"                               },
    { VK_TAB,    ShiftKey,           "InsertBacktab"                           },
    { VK_RETURN, 0,                  "InsertNewline"                           },
    { VK_RETURN, CtrlKey,            "InsertNewline"                           },
    { VK_RETURN, AltKey,             "InsertNewline"                           },
    { VK_RETURN, AltKey | ShiftKey,  "InsertNewline"                           },
    { VK_RETURN, ShiftKey,           "InsertLineBreak"                         },
    // These probably need handling somewhere else so do not execute them. The
    // 'Cut' command is removing text so let's avoid losing the user losing data
    // until we implement clipboard support wherever it should be.
    { VK_INSERT, CtrlKey,            "Copy"                                    },
    { VK_INSERT, ShiftKey,           "Paste"                                   },
    { VK_DELETE, ShiftKey,           "Cut"                                     },
    { 'C',         CtrlKey,          "Copy"                                    },
    { 'V',         CtrlKey,          "Paste"                                   },
    { 'V',         CtrlKey | ShiftKey, "PasteAndMatchStyle"                    },
    { 'X',         CtrlKey,          "Cut"                                     },
    { 'A',         CtrlKey,          "SelectAll"                               },
    { 'Z',         CtrlKey,          "Undo"                                    },
    { 'Z',         CtrlKey | ShiftKey, "Redo"                                  },
    { 'Y',         CtrlKey,          "Redo"                                    },
    { VK_INSERT, 0,                  "OverWrite"                               },
};

static const KeyPressEntry keyPressEntries[] = {
    { '\t',   0,                  "InsertTab"                                  },
    { '\t',   ShiftKey,           "InsertBacktab"                              },
    { '\r',   0,                  "InsertNewline"                              },
    { '\r',   CtrlKey,            "InsertNewline"                              },
    { '\r',   ShiftKey,           "InsertLineBreak"                            },
    { '\r',   AltKey,             "InsertNewline"                              },
    { '\r',   AltKey | ShiftKey,  "InsertNewline"                              },
};

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

static const char* interpretKeyEvent(const KeyboardEvent& event)
{
    static NeverDestroyed<HashMap<int, const char*>> keyDownCommandsMap;
    static NeverDestroyed<HashMap<int, const char*>> keyPressCommandsMap;

    if (keyDownCommandsMap.get().isEmpty()) {
        for (unsigned i = 0; i < std::size(keyDownEntries); i++)
            keyDownCommandsMap.get().set(keyDownEntries[i].modifiers << 16 | keyDownEntries[i].virtualKey, keyDownEntries[i].name);
        for (unsigned i = 0; i < std::size(keyPressEntries); i++)
            keyPressCommandsMap.get().set(keyPressEntries[i].modifiers << 16 | keyPressEntries[i].charCode, keyPressEntries[i].name);
    }

    unsigned modifiers = 0;
    if (event.shiftKey())
        modifiers |= ShiftKey;
    if (event.altKey())
        modifiers |= AltKey;
    if (event.ctrlKey())
        modifiers |= CtrlKey;
    if (event.metaKey())
        modifiers |= MetaKey;

    if (event.type() == eventNames().keydownEvent) {
        int mapKey = modifiers << 16 | event.keyCode();
        return mapKey ? keyDownCommandsMap.get().get(mapKey) : nullptr;
    }

    int mapKey = modifiers << 16 | event.charCode();
    return mapKey ? keyPressCommandsMap.get().get(mapKey) : nullptr;
}

static void handleKeyPress(LocalFrame& frame, KeyboardEvent& event, const PlatformKeyboardEvent& platformEvent)
{
    auto commandName = String::fromLatin1(interpretKeyEvent(event));

    if (!commandName.isEmpty()) {
        frame.editor().command(commandName).execute();
        event.setDefaultHandled();
        return;
    }

    // Don't insert null or control characters as they can result in unexpected behaviour
    if (event.charCode() < ' ')
        return;

    // Don't insert anything if a modifier is pressed and it has not been handled yet
    if (platformEvent.controlKey() || platformEvent.altKey())
        return;

    if (frame.editor().insertText(platformEvent.text(), &event))
        event.setDefaultHandled();
}

static void handleKeyDown(LocalFrame& frame, KeyboardEvent& event, const PlatformKeyboardEvent&)
{
    auto commandName = String::fromLatin1(interpretKeyEvent(event));
    if (commandName.isEmpty())
        return;

    // We shouldn't insert text through the editor. Let WebCore decide
    // how to handle that (say, Tab, which could also be used to
    // change focus).
    Editor::Command command = frame.editor().command(commandName);
    if (command.isTextInsertion())
        return;

    command.execute();
    event.setDefaultHandled();
}

void WebEditorClient::handleKeyboardEvent(WebCore::KeyboardEvent& event)
{
    ASSERT(event.target());
    auto* frame = downcast<Node>(event.target())->document().frame();
    ASSERT(frame);

    // FIXME: Reorder the checks in a more sensible way.

    auto* platformEvent = event.underlyingPlatformEvent();
    if (!platformEvent)
        return;

    // If this was an IME event don't do anything.
    if (platformEvent->windowsVirtualKeyCode() == VK_PROCESSKEY)
        return;

    // Don't allow text insertion for nodes that cannot edit.
    if (!frame->editor().canEdit())
        return;

    // This is just a normal text insertion, so wait to execute the insertion
    // until a keypress event happens. This will ensure that the insertion will not
    // be reflected in the contents of the field until the keyup DOM event.
    if (event.type() == eventNames().keypressEvent)
        return handleKeyPress(*frame, event, *platformEvent);
    if (event.type() == eventNames().keydownEvent)
        return handleKeyDown(*frame, event, *platformEvent);
}

} // namespace WebKit
