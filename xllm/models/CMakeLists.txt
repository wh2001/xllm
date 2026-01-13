include(cc_library)

set(MODELS_DEPS :model)
if(USE_NPU)
  list(APPEND MODELS_DEPS mf_smem)
endif()

# Define the library
cc_library(
  NAME
    models
  HDRS
    model_registry.h
    models.h
  SRCS
    model_registry.cpp
  DEPS
    ${MODELS_DEPS}
)
