/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#include "modules/desktop_capture/mac/full_screen_mac_application_handler.h"

#include <libproc.h>

#include <algorithm>
#include <functional>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "api/function_view.h"
#include "modules/desktop_capture/mac/window_list_utils.h"

namespace webrtc {
namespace {

static constexpr const char* kPowerPointSlideShowTitles[] = {
    "PowerPoint-BildschirmprÃ¤sentation",
    "Î ÏÎ¿Î²Î¿Î»Î® Ï€Î±ÏÎ¿Ï…ÏƒÎ¯Î±ÏƒÎ·Ï‚ PowerPoint",
    "PowerPoint ã‚¹ãƒ©ã‚¤ãƒ‰ ã‚·ãƒ§ãƒ¼",
    "PowerPoint Slide Show",
    "PowerPoint å¹»ç¯ç‰‡æ”¾æ˜ ",
    "PresentaciÃ³n de PowerPoint",
    "PowerPoint-slideshow",
    "Presentazione di PowerPoint",
    "PrezentÃ¡cia programu PowerPoint",
    "ApresentaÃ§Ã£o do PowerPoint",
    "PowerPoint-bildspel",
    "Prezentace v aplikaci PowerPoint",
    "PowerPoint ìŠ¬ë¼ì´ë“œ ì‡¼",
    "PowerPoint-lysbildefremvisning",
    "PowerPoint-vetÃ­tÃ©s",
    "PowerPoint Slayt GÃ¶sterisi",
    "Pokaz slajdÃ³w programu PowerPoint",
    "PowerPoint æŠ•å½±ç‰‡æ”¾æ˜ ",
    "Ð”ÐµÐ¼Ð¾Ð½ÑÑ‚Ñ€Ð°Ñ†Ð¸Ñ PowerPoint",
    "Diaporama PowerPoint",
    "PowerPoint-diaesitys",
    "Peragaan Slide PowerPoint",
    "PowerPoint-diavoorstelling",
    "à¸à¸²à¸£à¸™à¸³à¹€à¸ªà¸™à¸­à¸ªà¹„à¸¥à¸”à¹Œ PowerPoint",
    "ApresentaÃ§Ã£o de slides do PowerPoint",
    "×”×¦×’×ª ×©×§×•×¤×™×•×ª ×©×œ PowerPoint",
    "Ø¹Ø±Ø¶ Ø´Ø±Ø§Ø¦Ø­ ÙÙŠ PowerPoint"};

class FullScreenMacApplicationHandler : public FullScreenApplicationHandler {
 public:
  using TitlePredicate =
      std::function<bool(absl::string_view, absl::string_view)>;

  FullScreenMacApplicationHandler(DesktopCapturer::SourceId sourceId,
                                  TitlePredicate title_predicate,
                                  bool ignore_original_window)
      : FullScreenApplicationHandler(sourceId),
        title_predicate_(title_predicate),
        owner_pid_(GetWindowOwnerPid(sourceId)),
        ignore_original_window_(ignore_original_window) {}

 protected:
  using CachePredicate =
      rtc::FunctionView<bool(const DesktopCapturer::Source&)>;

  void InvalidateCacheIfNeeded(const DesktopCapturer::SourceList& source_list,
                               int64_t timestamp,
                               CachePredicate predicate) const {
    if (timestamp != cache_timestamp_) {
      cache_sources_.clear();
      std::copy_if(source_list.begin(), source_list.end(),
                   std::back_inserter(cache_sources_), predicate);
      cache_timestamp_ = timestamp;
    }
  }

  WindowId FindFullScreenWindowWithSamePid(
      const DesktopCapturer::SourceList& source_list,
      int64_t timestamp) const {
    InvalidateCacheIfNeeded(source_list, timestamp,
                            [&](const DesktopCapturer::Source& src) {
                              return src.id != GetSourceId() &&
                                     GetWindowOwnerPid(src.id) == owner_pid_;
                            });
    if (cache_sources_.empty())
      return kCGNullWindowID;

    const auto original_window = GetSourceId();
    const std::string title = GetWindowTitle(original_window);

    // We can ignore any windows with empty titles cause regardless type of
    // application it's impossible to verify that full screen window and
    // original window are related to the same document.
    if (title.empty())
      return kCGNullWindowID;

    MacDesktopConfiguration desktop_config =
        MacDesktopConfiguration::GetCurrent(
            MacDesktopConfiguration::TopLeftOrigin);

    const auto it = std::find_if(
        cache_sources_.begin(), cache_sources_.end(),
        [&](const DesktopCapturer::Source& src) {
          const std::string window_title = GetWindowTitle(src.id);

          if (window_title.empty())
            return false;

          if (title_predicate_ && !title_predicate_(title, window_title))
            return false;

          return IsWindowFullScreen(desktop_config, src.id);
        });

    return it != cache_sources_.end() ? it->id : 0;
  }

  DesktopCapturer::SourceId FindFullScreenWindow(
      const DesktopCapturer::SourceList& source_list,
      int64_t timestamp) const override {
    return !ignore_original_window_ && IsWindowOnScreen(GetSourceId())
               ? 0
               : FindFullScreenWindowWithSamePid(source_list, timestamp);
  }

 protected:
  const TitlePredicate title_predicate_;
  const int owner_pid_;
  const bool ignore_original_window_;
  mutable int64_t cache_timestamp_ = 0;
  mutable DesktopCapturer::SourceList cache_sources_;
};

bool equal_title_predicate(absl::string_view original_title,
                           absl::string_view title) {
  return original_title == title;
}

bool slide_show_title_predicate(absl::string_view original_title,
                                absl::string_view title) {
  if (title.find(original_title) == absl::string_view::npos)
    return false;

  for (const char* pp_slide_title : kPowerPointSlideShowTitles) {
    if (absl::StartsWith(title, pp_slide_title))
      return true;
  }
  return false;
}

class OpenOfficeApplicationHandler : public FullScreenMacApplicationHandler {
 public:
  OpenOfficeApplicationHandler(DesktopCapturer::SourceId sourceId)
      : FullScreenMacApplicationHandler(sourceId, nullptr, false) {}

  DesktopCapturer::SourceId FindFullScreenWindow(
      const DesktopCapturer::SourceList& source_list,
      int64_t timestamp) const override {
    InvalidateCacheIfNeeded(source_list, timestamp,
                            [&](const DesktopCapturer::Source& src) {
                              return GetWindowOwnerPid(src.id) == owner_pid_;
                            });

    const auto original_window = GetSourceId();
    const std::string original_title = GetWindowTitle(original_window);

    // Check if we have only one document window, otherwise it's not possible
    // to securely match a document window and a slide show window which has
    // empty title.
    if (std::any_of(cache_sources_.begin(), cache_sources_.end(),
                    [&original_title](const DesktopCapturer::Source& src) {
                      return src.title.length() && src.title != original_title;
                    })) {
      return kCGNullWindowID;
    }

    MacDesktopConfiguration desktop_config =
        MacDesktopConfiguration::GetCurrent(
            MacDesktopConfiguration::TopLeftOrigin);

    // Looking for slide show window,
    // it must be a full screen window with empty title
    const auto slide_show_window = std::find_if(
        cache_sources_.begin(), cache_sources_.end(), [&](const auto& src) {
          return src.title.empty() &&
                 IsWindowFullScreen(desktop_config, src.id);
        });

    if (slide_show_window == cache_sources_.end()) {
      return kCGNullWindowID;
    }

    return slide_show_window->id;
  }
};

}  // namespace

std::unique_ptr<FullScreenApplicationHandler>
CreateFullScreenMacApplicationHandler(DesktopCapturer::SourceId sourceId) {
  std::unique_ptr<FullScreenApplicationHandler> result;
  int pid = GetWindowOwnerPid(sourceId);
  char buffer[PROC_PIDPATHINFO_MAXSIZE];
  int path_length = proc_pidpath(pid, buffer, sizeof(buffer));
  if (path_length > 0) {
    const char* last_slash = strrchr(buffer, '/');
    const std::string name{last_slash ? last_slash + 1 : buffer};
    const std::string owner_name = GetWindowOwnerName(sourceId);
    FullScreenMacApplicationHandler::TitlePredicate predicate = nullptr;
    bool ignore_original_window = false;
    if (name.find("Google Chrome") == 0 || name == "Chromium") {
      predicate = equal_title_predicate;
    } else if (name == "Microsoft PowerPoint") {
      predicate = slide_show_title_predicate;
      ignore_original_window = true;
    } else if (name == "Keynote") {
      predicate = equal_title_predicate;
    } else if (owner_name == "OpenOffice") {
      return std::make_unique<OpenOfficeApplicationHandler>(sourceId);
    }

    if (predicate) {
      result.reset(new FullScreenMacApplicationHandler(sourceId, predicate,
                                                       ignore_original_window));
    }
  }

  return result;
}

}  // namespace webrtc
