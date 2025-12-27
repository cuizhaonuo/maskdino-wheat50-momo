import contextvars
indices_ctx = contextvars.ContextVar('indices')
draw_encoder_class_vars_ctx = contextvars.ContextVar('draw_encoder_class_vars')
attn_output_weights_ctx = contextvars.ContextVar('attn_output_weights')
stacked_attn_output_weights_ctx = contextvars.ContextVar('stacked_attn_output_weights')
layer_query_ids_ctx = contextvars.ContextVar('layer_query_ids')