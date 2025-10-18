#pragma once

#include <adwaita.h>

G_BEGIN_DECLS

typedef enum {
  PMS_WINDOW_INITIAL_STATE_EMPTY,
  PMS_WINDOW_INITIAL_STATE_SCREENCAST,
  PMS_WINDOW_INITIAL_STATE_CAMERA,
} PmsWindowInitialState;

#define PMS_TYPE_WINDOW (pms_window_get_type())
G_DECLARE_FINAL_TYPE (PmsWindow, pms_window, PMS, WINDOW, AdwApplicationWindow)

G_END_DECLS
