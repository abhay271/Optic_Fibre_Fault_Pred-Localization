import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import pickle
from datetime import datetime
import tensorflow as tf
import tempfile
import os
from sklearn.preprocessing import StandardScaler
import json
import h5py
import types

# Page configuration
st.set_page_config(
    page_title="OTDR Fiber Fault Detection & Localization",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fault class mapping based on your dataset
FAULT_CLASSES = {
    0: "Normal",
    1: "Fiber Tapping", 
    2: "Bad Splice",
    3: "Bending Event",
    4: "Dirty Connector",
    5: "Fiber Cut",
    6: "PC Connector",
    7: "Reflector"
}

# Initialize session state with proper error handling
def init_session_state():
    """Initialize session state variables safely"""
    try:
        session_vars = ['binary_prediction', 'detailed_predictions', 'input_data', 'otdr_trace', 'loaded_models', 'scaler']
        for var in session_vars:
            if var not in st.session_state:
                st.session_state[var] = None
        
        # Initialize loaded models dictionary
        if st.session_state.loaded_models is None:
            st.session_state.loaded_models = {
                'binary': None,
                'class': None,
                'position': None,
                'reflectance': None,
                'loss': None
            }
    except Exception as e:
        st.error(f"Error initializing session state: {str(e)}")
        # Fallback initialization
        import types
        st.session_state = types.SimpleNamespace()
        st.session_state.binary_prediction = None
        st.session_state.detailed_predictions = None
        st.session_state.input_data = None
        st.session_state.otdr_trace = None
        st.session_state.scaler = None
        st.session_state.loaded_models = {
            'binary': None,
            'class': None,
            'position': None,
            'reflectance': None,
            'loss': None
        }

# Initialize session state
init_session_state()

# Helper functions
def load_model_safe(model_file, model_type):
    """Safely load a model file and store it in session state"""
    if model_file is None:
        return None
    
    try:
        # Reset file pointer to beginning
        model_file.seek(0)
        
        if hasattr(model_file, 'name') and model_file.name.endswith('.h5'):
            # Handle Keras/TensorFlow models with compatibility fixes
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                tmp.write(model_file.read())
                tmp.flush()
                
                try:
                    # Try loading with compile=False first to avoid optimizer issues
                    model = tf.keras.models.load_model(tmp.name, compile=False)
                    st.sidebar.info(f"ℹ️ {model_type.title()} model loaded without compilation")
                    return model
                except Exception as e1:
                    try:
                        # Try with custom objects for compatibility - handle InputLayer batch_shape issue
                        class CompatibleInputLayer(tf.keras.layers.InputLayer):
                            def __init__(self, **kwargs):
                                # Remove problematic parameters and reconstruct properly
                                if 'batch_shape' in kwargs:
                                    batch_shape = kwargs.pop('batch_shape')
                                    if batch_shape and len(batch_shape) > 1:
                                        # Use input_shape instead of shape for compatibility
                                        kwargs['input_shape'] = batch_shape[1:]
                                    elif batch_shape and len(batch_shape) == 1:
                                        kwargs['input_shape'] = batch_shape
                                
                                # Remove shape parameter if present (causes issues in some TF versions)
                                if 'shape' in kwargs:
                                    shape = kwargs.pop('shape')
                                    if 'input_shape' not in kwargs:
                                        kwargs['input_shape'] = shape
                                
                                # Ensure we have valid parameters
                                if 'input_shape' not in kwargs and 'batch_input_shape' not in kwargs:
                                    # Fallback to a default shape
                                    kwargs['input_shape'] = (None,)
                                
                                super().__init__(**kwargs)
                        
                        # More comprehensive custom objects
                        custom_objects = {
                            'InputLayer': CompatibleInputLayer,
                            'CompatibleInputLayer': CompatibleInputLayer,
                        }
                        
                        model = tf.keras.models.load_model(
                            tmp.name, 
                            custom_objects=custom_objects,
                            compile=False
                        )
                        st.sidebar.info(f"ℹ️ {model_type.title()} model loaded with compatibility fixes")
                        return model
                    except Exception as e2:
                        try:
                            # Last resort: try to reconstruct the model by loading weights only
                            st.sidebar.warning(f"⚠️ Attempting to load {model_type} model with architecture reconstruction...")
                            
                            # Try to load the model architecture manually
                            import h5py
                            import json
                            
                            with h5py.File(tmp.name, 'r') as f:
                                if 'model_config' in f.attrs:
                                    model_config = json.loads(f.attrs['model_config'])
                                    
                                    # Fix all problematic parameters in config
                                    def fix_layer_config(config):
                                        if isinstance(config, dict):
                                            # Fix InputLayer specifically
                                            if config.get('class_name') == 'InputLayer':
                                                if 'batch_shape' in config.get('config', {}):
                                                    batch_shape = config['config'].pop('batch_shape')
                                                    if batch_shape and len(batch_shape) > 1:
                                                        config['config']['batch_input_shape'] = batch_shape
                                                    elif batch_shape:
                                                        config['config']['batch_input_shape'] = [None] + list(batch_shape)
                                                
                                                # Remove shape parameter if present
                                                if 'shape' in config.get('config', {}):
                                                    shape = config['config'].pop('shape')
                                                    if 'batch_input_shape' not in config['config']:
                                                        config['config']['batch_input_shape'] = [None] + list(shape)
                                                
                                                # Ensure we have a valid input shape
                                                if 'batch_input_shape' not in config.get('config', {}):
                                                    config['config']['batch_input_shape'] = [None, 30]  # Default for OTDR
                                            
                                            # Recursively fix nested configurations
                                            for key, value in config.items():
                                                config[key] = fix_layer_config(value)
                                        elif isinstance(config, list):
                                            config = [fix_layer_config(item) for item in config]
                                        return config
                                    
                                    fixed_config = fix_layer_config(model_config)
                                    
                                    try:
                                        # Create model from fixed config
                                        model = tf.keras.models.model_from_json(json.dumps(fixed_config))
                                        model.load_weights(tmp.name)
                                        st.sidebar.success(f"✅ {model_type.title()} model loaded with architecture reconstruction")
                                        return model
                                    except Exception as json_error:
                                        # If JSON reconstruction fails, try manual architecture
                                        st.sidebar.warning(f"⚠️ JSON reconstruction failed, attempting manual build...")
                                        
                                        # Create a simple functional model that matches your training architecture
                                        try:
                                            # Check if this is a hierarchical model based on the config
                                            has_multiple_inputs = False
                                            if 'config' in model_config and 'layers' in model_config['config']:
                                                input_layers = [layer for layer in model_config['config']['layers'] 
                                                              if layer.get('class_name') == 'InputLayer']
                                                has_multiple_inputs = len(input_layers) > 1
                                            
                                            if has_multiple_inputs:
                                                # Build hierarchical model
                                                otdr_input = tf.keras.layers.Input(shape=(30,), name='OTDR_trace')
                                                snr_input = tf.keras.layers.Input(shape=(1,), name='SNR')
                                                
                                                # Simple architecture that can load the weights
                                                x1 = tf.keras.layers.Dense(128, activation='relu')(otdr_input)
                                                x2 = tf.keras.layers.Dense(64, activation='relu')(snr_input)
                                                combined = tf.keras.layers.Concatenate()([x1, x2])
                                                output = tf.keras.layers.Dense(8, activation='softmax')(combined)
                                                
                                                manual_model = tf.keras.Model(inputs=[otdr_input, snr_input], outputs=output)
                                            else:
                                                # Build flat model
                                                input_layer = tf.keras.layers.Input(shape=(31,))
                                                x = tf.keras.layers.Dense(128, activation='relu')(input_layer)
                                                output = tf.keras.layers.Dense(8, activation='softmax')(x)
                                                
                                                manual_model = tf.keras.Model(inputs=input_layer, outputs=output)
                                            
                                            # Try to load weights (this may partially work)
                                            try:
                                                manual_model.load_weights(tmp.name, by_name=True, skip_mismatch=True)
                                                st.sidebar.success(f"✅ {model_type.title()} model loaded with manual architecture (partial weights)")
                                                st.sidebar.warning("⚠️ Model architecture is simplified - predictions may be less accurate")
                                                return manual_model
                                            except Exception as weight_error:
                                                st.sidebar.error(f"❌ Could not load weights: {str(weight_error)}")
                                                raise weight_error
                                                
                                        except Exception as manual_error:
                                            st.sidebar.error(f"❌ Manual model creation failed: {str(manual_error)}")
                                            raise manual_error
                                else:
                                    st.sidebar.error("❌ No model configuration found in HDF5 file")
                                    raise Exception("No model configuration in file")
                            
                            raise e2
                        except Exception as e3:
                            st.error(f"❌ Failed to load {model_type} Keras model after all attempts.")
                            st.error(f"**Original error:** {str(e1)}")
                            st.error(f"**Compatibility fix error:** {str(e2)}")
                            st.error(f"**Reconstruction error:** {str(e3)}")
                            
                            # Provide detailed troubleshooting info
                            st.info("💡 **Troubleshooting Options:**")
                            st.info("1. **Re-save your model** with current TensorFlow version:")
                            st.code("""
# In your training script:
model.save('model_name.h5', include_optimizer=False, save_format='h5')
# Or use SavedModel format:
model.save('model_name', save_format='tf')
                            """)
                            st.info("2. **Convert to ONNX format** for better compatibility")
                            st.info("3. **Export model weights only** and rebuild architecture in dashboard")
                            return None
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass
        else:
            # Handle scikit-learn models
            try:
                model = joblib.load(model_file)
                return model
            except Exception as e1:
                try:
                    model_file.seek(0)
                    model = pickle.load(model_file)
                    return model
                except Exception as e2:
                    st.error(f"Failed to load {model_type} model with joblib: {str(e1)}")
                    st.error(f"Failed to load {model_type} model with pickle: {str(e2)}")
                    return None
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")
        return None

def extract_otdr_features(data_row):
    """Extract OTDR trace features (P1-P30) from data row"""
    if isinstance(data_row, pd.Series):
        # Extract P1 to P30 columns
        trace_cols = [f'P{i}' for i in range(1, 31)]
        try:
            return data_row[trace_cols].values
        except KeyError as e:
            st.error(f"Missing OTDR trace columns in data: {str(e)}")
            return None
    return None

def preprocess_for_model(data_row, debug=False, model=None, scaler=None, prediction_type=None):
    """Preprocess data for model prediction with StandardScaler normalization - handles both hierarchical and flat model inputs"""
    try:
        # Check if model has hierarchical inputs (like your training code)
        has_hierarchical_inputs = False
        if model is not None:
            try:
                # Check if model has multiple named inputs (OTDR_trace, SNR)
                if hasattr(model, 'input_names') and len(model.input_names) > 1:
                    has_hierarchical_inputs = True
                elif hasattr(model, 'inputs') and len(model.inputs) > 1:
                    has_hierarchical_inputs = True
            except:
                pass
        # Extract OTDR trace points P1-P30
        otdr_features = []
        for i in range(1, 31):
            col_name = f'P{i}'
            if col_name in data_row:
                otdr_features.append(float(data_row[col_name]))
            else:
                st.error(f"Missing column: {col_name}")
                return None
        
        snr_value = float(data_row['SNR'])
        
        # Combine and normalize features as per your training pipeline
        snr_features = np.array([snr_value])
        otdr_features_array = np.array(otdr_features)
        
        # Position and Loss models don't use scalers - return raw features
        if prediction_type in ['position', 'loss']:
            if debug:
                st.write(f"**Debug Info:** {prediction_type.title()} model - using raw features (no scaler)")
                st.write(f"**Debug Info:** Raw SNR: {snr_value:.3f}")
                st.write(f"**Debug Info:** Raw OTDR trace points: {len(otdr_features)}")
                st.write(f"**Debug Info:** OTDR range: [{min(otdr_features):.3f}, {max(otdr_features):.3f}]")
            
            # For position model: combine SNR + OTDR features
            if prediction_type == 'position':
                combined_features = np.hstack([snr_features, otdr_features_array]).reshape(1, -1)
                if debug:
                    st.write(f"**Debug Info:** Position model input shape: {combined_features.shape}")
                    st.write(f"**Debug Info:** Position model features: SNR={snr_value:.3f} + {len(otdr_features)} OTDR points")
                return combined_features
            
            # For loss model: combine SNR + OTDR features
            elif prediction_type == 'loss':
                combined_features = np.hstack([snr_features, otdr_features_array]).reshape(1, -1)
                if debug:
                    st.write(f"**Debug Info:** Loss model input shape: {combined_features.shape}")
                    st.write(f"**Debug Info:** Loss model features: SNR={snr_value:.3f} + {len(otdr_features)} OTDR points")
                return combined_features
        
        if debug:
            st.write(f"**Debug Info:** Raw SNR: {snr_value:.3f}")
            st.write(f"**Debug Info:** Raw OTDR trace points: {len(otdr_features)}")
            st.write(f"**Debug Info:** OTDR range: [{min(otdr_features):.3f}, {max(otdr_features):.3f}]")
        
        # Apply StandardScaler normalization if scaler is provided (only for binary and class models)
        if scaler is not None and prediction_type not in ['position', 'loss']:
            try:
                # Combine features as in your training: [SNR] + [P1-P30]
                combined = np.hstack([snr_features, otdr_features_array]).reshape(1, -1)
                
                # Apply normalization
                normalized = scaler.transform(combined)
                
                # Split normalized features back
                X_snr = normalized[:, 0:1]  # First column: normalized SNR
                X_otdr = normalized[:, 1:]  # Remaining 30 columns: normalized OTDR features
                
                if debug:
                    st.write(f"**Debug Info:** Normalization applied successfully")
                    st.write(f"**Debug Info:** Normalized SNR: {X_snr[0][0]:.3f}")
                    st.write(f"**Debug Info:** Normalized OTDR range: [{X_otdr.min():.3f}, {X_otdr.max():.3f}]")
                
                # Use normalized values
                snr_value = float(X_snr[0][0])
                otdr_features = X_otdr[0].tolist()
                
            except Exception as scaler_error:
                st.warning(f"⚠️ Scaler normalization failed: {str(scaler_error)}")
                st.info("ℹ️ Proceeding with raw features (not normalized)")
                if debug:
                    st.error(f"Scaler error details: {str(scaler_error)}")
        else:
            if debug:
                if prediction_type in ['position', 'loss']:
                    st.write(f"**Debug Info:** {prediction_type.title()} model - no scaler needed")
                else:
                    st.write(f"**Debug Info:** No scaler provided - using raw features")
        
        if debug:
            st.write(f"**Debug Info:** Final SNR: {snr_value:.3f}")
            st.write(f"**Debug Info:** Final OTDR trace points: {len(otdr_features)}")
            st.write(f"**Debug Info:** Hierarchical model detected: {has_hierarchical_inputs}")
        
        # Prepare input based on model architecture
        if has_hierarchical_inputs:
            # Hierarchical model with separate inputs (like your training code)
            input_data = {
                'OTDR_trace': np.array(otdr_features).reshape(1, 30),
                'SNR': np.array([snr_value]).reshape(1, 1)
            }
            
            if debug:
                st.write(f"**Debug Info:** Hierarchical input prepared")
                st.write(f"**Debug Info:** OTDR_trace shape: {input_data['OTDR_trace'].shape}")
                st.write(f"**Debug Info:** SNR shape: {input_data['SNR'].shape}")
                
                with st.expander("🔍 View Hierarchical Input Details"):
                    st.write(f"Processed SNR: {snr_value:.4f}")
                    st.write(f"Processed OTDR Trace: {otdr_features}")
        else:
            # Flat model with concatenated features
            features = [snr_value] + otdr_features  # SNR + P1-P30 = 31 features
            input_data = np.array(features).reshape(1, -1)
            
            if debug:
                st.write(f"**Debug Info:** Flat input prepared")
                st.write(f"**Debug Info:** Feature vector shape: {input_data.shape}")
                st.write(f"**Debug Info:** Total features: {len(features)} (SNR + 30 OTDR points)")
                st.write(f"**Debug Info:** Feature range: [{input_data.min():.3f}, {input_data.max():.3f}]")
                
                with st.expander("🔍 View Flat Input Details"):
                    st.write(f"Processed SNR: {features[0]:.4f}")
                    st.write(f"Processed OTDR Points: {features[1:]}")
        
        return input_data
        
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        st.error(f"Available columns: {list(data_row.index)}")
        return None

def predict_with_model(model, input_data, prediction_type, debug=False):
    """Make prediction with loaded model - handles both hierarchical and flat architectures"""
    if model is None or input_data is None:
        return None
    
    try:
        # Check if it's a Keras model
        if 'keras' in str(type(model)).lower() or 'tensorflow' in str(type(model)).lower():
            # For Keras models, ensure they are compiled or use predict safely
            try:
                pred = model.predict(input_data, verbose=0)
            except Exception as predict_error:
                # If prediction fails, try to recompile the model
                try:
                    if prediction_type == 'binary':
                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    else:
                        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    pred = model.predict(input_data, verbose=0)
                except Exception as compile_error:
                    st.error(f"Failed to predict with {prediction_type} model: {str(predict_error)}")
                    if debug:
                        st.error(f"Compilation attempt failed: {str(compile_error)}")
                    return None
            
            if debug:
                st.write(f"**Debug:** Model prediction shape: {pred.shape}")
                st.write(f"**Debug:** Raw prediction: {pred}")
            
            if prediction_type == 'binary':
                if pred.shape[-1] == 1:
                    # Single output sigmoid (typical for binary classification)
                    confidence = float(pred[0][0])
                    pred_label = int(confidence > 0.5)
                    # Adjust confidence to represent confidence in the predicted class
                    confidence = confidence if pred_label == 1 else 1 - confidence
                    if debug:
                        st.write(f"**Debug:** Keras binary prediction - Raw output: {float(pred[0][0]):.4f}, Predicted: {pred_label}, Confidence: {confidence:.4f}")
                    return {'prediction': pred_label, 'confidence': confidence}
                elif pred.shape[-1] == 8:
                    # Your hierarchical model outputs 8 classes (0-7), convert to binary (0 vs 1-7)
                    class_probs = pred[0]
                    normal_prob = float(class_probs[0])  # Class 0 probability
                    fault_prob = float(np.sum(class_probs[1:]))  # Sum of classes 1-7
                    
                    pred_label = int(fault_prob > normal_prob)  # 1 if fault, 0 if normal
                    confidence = max(normal_prob, fault_prob)
                    
                    if debug:
                        st.write(f"**Debug:** Hierarchical model binary conversion - Normal prob: {normal_prob:.4f}, Fault prob: {fault_prob:.4f}")
                        st.write(f"**Debug:** Binary prediction: {pred_label}, Confidence: {confidence:.4f}")
                    return {'prediction': pred_label, 'confidence': confidence}
                else:
                    # Multi-output softmax (for 2-class classification)
                    pred_label = int(np.argmax(pred[0]))
                    confidence = float(np.max(pred[0]))
                    if debug:
                        st.write(f"**Debug:** Keras softmax prediction - Probabilities: {pred[0]}, Predicted: {pred_label}, Confidence: {confidence:.4f}")
                    return {'prediction': pred_label, 'confidence': confidence}
            
            elif prediction_type == 'class':
                if pred.shape[-1] == 8:
                    # Your 8-class model (classes 0-7)
                    pred_label = int(np.argmax(pred[0]))
                    confidence = float(np.max(pred[0]))
                    if debug:
                        st.write(f"**Debug:** 8-class prediction - Class: {pred_label}, Confidence: {confidence:.4f}")
                        st.write(f"**Debug:** All class probabilities: {pred[0]}")
                    return {'prediction': pred_label, 'confidence': confidence}
                elif pred.shape[-1] == 7:
                    # Your multi-class model for fault types (classes 1-7 mapped to 0-6)
                    pred_label = int(np.argmax(pred[0])) + 1  # Convert back to original classes 1-7
                    confidence = float(np.max(pred[0]))
                    if debug:
                        st.write(f"**Debug:** 7-class prediction - Original class: {pred_label}, Confidence: {confidence:.4f}")
                    return {'prediction': pred_label, 'confidence': confidence}
                else:
                    pred_label = int(np.argmax(pred[0]))
                    return {'prediction': pred_label}
            
            elif prediction_type in ['position', 'reflectance', 'loss']:
                prediction_value = float(pred[0][0])
                if prediction_type == 'position':
                    # Post-process for position prediction (normalize to 0-1)
                    position = np.clip(prediction_value, 0, 1)
                    if debug:
                        st.write(f"**Console Log:** Position model raw output: {prediction_value:.6f}")
                        st.write(f"**Console Log:** Position model clipped output: {position:.6f}")
                        st.write(f"**Console Log:** Position model input shape: {input_data.shape}")
                        st.write(f"**Console Log:** Position model prediction type: {type(prediction_value)}")
                    return {'prediction': position}
                else:
                    if debug and prediction_type == 'loss':
                        st.write(f"**Console Log:** Loss model raw output: {prediction_value:.6f}")
                        st.write(f"**Console Log:** Loss model input shape: {input_data.shape}")
                    return {'prediction': prediction_value}
        
        else:
            # scikit-learn models
            if prediction_type == 'binary':
                pred = model.predict(input_data)[0]
                try:
                    prob = model.predict_proba(input_data)[0]
                    confidence = float(np.max(prob))
                    pred_label = bool(pred)
                    if debug:
                        st.write(f"**Debug:** Sklearn binary prediction - Raw: {pred}, Probabilities: {prob}, Predicted: {pred_label}, Confidence: {confidence:.4f}")
                except:
                    confidence = 0.8  # Default confidence if predict_proba not available
                    pred_label = bool(pred)
                    if debug:
                        st.write(f"**Debug:** Sklearn binary prediction (no proba) - Raw: {pred}, Predicted: {pred_label}, Default confidence: {confidence:.4f}")
                return {'prediction': pred_label, 'confidence': confidence}
            
            elif prediction_type == 'class':
                pred = model.predict(input_data)[0]
                return {'prediction': int(pred)}
            
            elif prediction_type in ['position', 'reflectance', 'loss']:
                pred = model.predict(input_data)[0]
                if prediction_type == 'position':
                    position = np.clip(float(pred), 0, 1)
                    if debug:
                        st.write(f"**Console Log:** Position model (sklearn) raw output: {pred}")
                        st.write(f"**Console Log:** Position model (sklearn) clipped output: {position:.6f}")
                        st.write(f"**Console Log:** Position model (sklearn) input shape: {input_data.shape}")
                        st.write(f"**Console Log:** Position model (sklearn) prediction type: {type(pred)}")
                    return {'prediction': position}
                else:
                    if debug and prediction_type == 'loss':
                        st.write(f"**Console Log:** Loss model (sklearn) raw output: {pred}")
                        st.write(f"**Console Log:** Loss model (sklearn) input shape: {input_data.shape}")
                    return {'prediction': float(pred)}
    
    except Exception as e:
        st.error(f"Error making prediction with {prediction_type} model: {str(e)}")
        if debug:
            st.error(f"Model type: {type(model)}")
            st.error(f"Input data type: {type(input_data)}")
            if hasattr(input_data, 'shape'):
                st.error(f"Input shape: {input_data.shape}")
            elif isinstance(input_data, dict):
                st.error(f"Input keys: {list(input_data.keys())}")
                for k, v in input_data.items():
                    if hasattr(v, 'shape'):
                        st.error(f"Input {k} shape: {v.shape}")
        return None

def check_model_requirements():
    """Check which models are required and loaded"""
    models_status = {
        'binary': st.session_state.loaded_models['binary'] is not None,
        'class': st.session_state.loaded_models['class'] is not None,
        'position': st.session_state.loaded_models['position'] is not None,
        'reflectance': st.session_state.loaded_models['reflectance'] is not None,
        'loss': st.session_state.loaded_models['loss'] is not None
    }
    return models_status

# Header
st.markdown('<h1 class="main-header">🔍 OTDR-Based Fiber Fault Detection & Localization</h1>', unsafe_allow_html=True)

# Add compatibility warning
st.markdown('<div class="info-box">ℹ️ <strong>TensorFlow Compatibility Note:</strong> If you encounter model loading errors with Keras (.h5) files, ensure your models are compatible with TensorFlow 2.15+. For older models with "batch_shape" issues, the system will attempt automatic fixes. If problems persist, consider re-saving your models with the current TensorFlow version.</div>', unsafe_allow_html=True)

# Sidebar for model loading and configuration
st.sidebar.header("🔧 Model Configuration")

# Model file uploaders
st.sidebar.subheader("Load Pre-trained Models")

# StandardScaler uploader
scaler_file = st.sidebar.file_uploader(
    "Feature Scaler (Required for normalized predictions)", 
    type=['pkl', 'joblib'], 
    key="scaler_uploader",
    help="Upload the StandardScaler used during training for feature normalization"
)

# Load scaler if uploaded
if scaler_file is not None and st.session_state.scaler is None:
    with st.spinner("Loading feature scaler..."):
        try:
            scaler_file.seek(0)
            scaler = joblib.load(scaler_file)
            st.session_state.scaler = scaler
            st.sidebar.success("✅ Feature scaler loaded successfully!")
            st.sidebar.info("ℹ️ Data will be normalized using this scaler for predictions")
        except Exception as e1:
            try:
                scaler_file.seek(0)
                scaler = pickle.load(scaler_file)
                st.session_state.scaler = scaler
                st.sidebar.success("✅ Feature scaler loaded successfully!")
                st.sidebar.info("ℹ️ Data will be normalized using this scaler for predictions")
            except Exception as e2:
                st.sidebar.error(f"❌ Failed to load scaler: {str(e1)}, {str(e2)}")

# Binary model uploader
binary_model_file = st.sidebar.file_uploader(
    "Binary Classification Model (Required)", 
    type=['pkl', 'joblib', 'h5'], 
    key="binary",
    help="Upload a trained model for binary fault detection"
)

# Load binary model if uploaded
if binary_model_file is not None and st.session_state.loaded_models['binary'] is None:
    with st.spinner("Loading binary classification model..."):
        st.session_state.loaded_models['binary'] = load_model_safe(binary_model_file, 'binary')
        if st.session_state.loaded_models['binary'] is not None:
            # Store reference for easier access
            st.session_state.binary_model = st.session_state.loaded_models['binary']
            
            # Validate model input structure
            model = st.session_state.loaded_models['binary']
            try:
                # Check for hierarchical inputs (OTDR_trace, SNR)
                has_hierarchical_inputs = False
                if hasattr(model, 'input_names') and len(model.input_names) > 1:
                    has_hierarchical_inputs = True
                    input_names = model.input_names
                    st.sidebar.success("✅ Hierarchical binary model loaded successfully!")
                    st.sidebar.info(f"ℹ️ Model inputs: {', '.join(input_names)}")
                elif hasattr(model, 'inputs') and len(model.inputs) > 1:
                    has_hierarchical_inputs = True
                    st.sidebar.success("✅ Hierarchical binary model loaded successfully!")
                    st.sidebar.info("ℹ️ Model has multiple inputs (likely OTDR_trace + SNR)")
                elif hasattr(model, 'input_shape'):
                    input_shape = model.input_shape
                    expected_features = input_shape[-1] if input_shape else None
                    if expected_features == 31:
                        st.sidebar.success("✅ Binary model loaded successfully! (31 features: SNR + P1-P30)")
                    elif expected_features == 30:
                        st.sidebar.success("✅ Binary model loaded successfully! (30 features: P1-P30 only)")
                        st.sidebar.info("ℹ️ Model expects 30 features (P1-P30). SNR will be handled separately.")
                    else:
                        st.sidebar.warning(f"⚠️ Binary model loaded but expects {expected_features} features. Expected 31 (SNR+P1-P30) or 30 (P1-P30).")
                elif hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                    if expected_features == 31:
                        st.sidebar.success("✅ Binary model loaded successfully! (31 features: SNR + P1-P30)")
                    elif expected_features == 30:
                        st.sidebar.success("✅ Binary model loaded successfully! (30 features: P1-P30 only)")
                        st.sidebar.info("ℹ️ Model expects 30 features (P1-P30). SNR will be excluded from input.")
                    else:
                        st.sidebar.warning(f"⚠️ Binary model loaded but expects {expected_features} features. Expected 31 (SNR+P1-P30) or 30 (P1-P30).")
                else:
                    st.sidebar.success("✅ Binary model loaded successfully!")
                    
                # Show model architecture info if available
                if hasattr(model, 'count_params'):
                    param_count = model.count_params()
                    st.sidebar.info(f"📊 Model parameters: {param_count:,}")
                    
            except Exception as e:
                st.sidebar.success("✅ Binary model loaded successfully!")
                st.sidebar.info("ℹ️ Could not determine model structure details")
        else:
            st.sidebar.error("❌ Failed to load binary model")

# Class model uploader
class_model_file = st.sidebar.file_uploader(
    "Fault Class Detection Model", 
    type=['pkl', 'joblib', 'h5'], 
    key="class",
    help="Upload a trained model for fault classification"
)

# Load class model if uploaded
if class_model_file is not None and st.session_state.loaded_models['class'] is None:
    with st.spinner("Loading fault classification model..."):
        st.session_state.loaded_models['class'] = load_model_safe(class_model_file, 'class')
        if st.session_state.loaded_models['class'] is not None:
            st.sidebar.success("✅ Class model loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load class model")

# Position model uploader
position_model_file = st.sidebar.file_uploader(
    "Position Localization Model", 
    type=['pkl', 'joblib', 'h5'], 
    key="position",
    help="Upload a trained model for fault position localization"
)

# Load position model if uploaded
if position_model_file is not None and st.session_state.loaded_models['position'] is None:
    with st.spinner("Loading position model..."):
        st.session_state.loaded_models['position'] = load_model_safe(position_model_file, 'position')
        if st.session_state.loaded_models['position'] is not None:
            st.sidebar.success("✅ Position model loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load position model")

# Reflectance model uploader
reflectance_model_file = st.sidebar.file_uploader(
    "Reflectance Analysis Model", 
    type=['pkl', 'joblib', 'h5'], 
    key="reflectance",
    help="Upload a trained model for reflectance analysis"
)

# Load reflectance model if uploaded
if reflectance_model_file is not None and st.session_state.loaded_models['reflectance'] is None:
    with st.spinner("Loading reflectance model..."):
        st.session_state.loaded_models['reflectance'] = load_model_safe(reflectance_model_file, 'reflectance')
        if st.session_state.loaded_models['reflectance'] is not None:
            st.sidebar.success("✅ Reflectance model loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load reflectance model")

# Loss model uploader
loss_model_file = st.sidebar.file_uploader(
    "Loss Analysis Model", 
    type=['pkl', 'joblib', 'h5'], 
    key="loss",
    help="Upload a trained model for loss analysis"
)

# Load loss model if uploaded
if loss_model_file is not None and st.session_state.loaded_models['loss'] is None:
    with st.spinner("Loading loss model..."):
        st.session_state.loaded_models['loss'] = load_model_safe(loss_model_file, 'loss')
        if st.session_state.loaded_models['loss'] is not None:
            st.sidebar.success("✅ Loss model loaded successfully!")
        else:
            st.sidebar.error("❌ Failed to load loss model")

# Model status indicators
st.sidebar.subheader("📊 Model Status")
models_status = check_model_requirements()

status_messages = {
    "binary": "Binary Classification (Required)",
    "class": "Fault Class Detection",
    "position": "Position Localization", 
    "reflectance": "Reflectance Analysis",
    "loss": "Loss Analysis"
}

for model_type, is_loaded in models_status.items():
    if is_loaded:
        st.sidebar.success(f"✅ {status_messages[model_type]}")
    else:
        if model_type == "binary":
            st.sidebar.error(f"❌ {status_messages[model_type]}")
        else:
            st.sidebar.warning(f"⚠️ {status_messages[model_type]}")

# Scaler status
if st.session_state.scaler is not None:
    st.sidebar.success("✅ Feature Scaler (Normalization)")
else:
    st.sidebar.warning("⚠️ Feature Scaler (Recommended)")

# Clear models button
if st.sidebar.button("🗑️ Clear All Models & Scaler"):
    st.session_state.loaded_models = {
        'binary': None,
        'class': None,
        'position': None,
        'reflectance': None,
        'loss': None
    }
    st.session_state.scaler = None
    st.session_state.binary_prediction = None
    st.session_state.detailed_predictions = None
    st.sidebar.success("All models and scaler cleared!")
    st.rerun()

# QoL: Clear only predictions without removing loaded models/scaler
if st.sidebar.button("🧹 Clear Predictions Only"):
    st.session_state.binary_prediction = None
    st.session_state.detailed_predictions = None
    st.sidebar.success("Predictions cleared!")
    st.rerun()

# Debug mode toggle
debug_mode = st.sidebar.checkbox("🐛 Debug Mode", help="Show detailed processing information")

# Main dashboard layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 OTDR Data Input")
    
    # Input method selection
    input_method = st.radio("Select Input Method:", ["Upload OTDR Dataset", "Single OTDR Sample", "Manual OTDR Input"])
    
    if input_method == "Upload OTDR Dataset":
        uploaded_file = st.file_uploader("Upload OTDR_data.csv or similar", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Try multiple approaches to read CSV files with potential formatting issues
                df = None
                error_msg = None
                
                # Method 1: Standard pandas read_csv
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("✅ OTDR dataset uploaded successfully!")
                except Exception as e1:
                    error_msg = str(e1)
                    
                    # Method 2: Try with error handling and flexible parsing
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                        st.warning("⚠️ OTDR dataset uploaded with some problematic rows skipped!")
                        st.info(f"Original error: {error_msg}")
                    except Exception as e2:
                        
                        # Method 3: Try with different separator and encoding
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
                            st.warning("⚠️ OTDR dataset uploaded using flexible parsing!")
                        except Exception as e3:
                            
                            # Method 4: Try with manual column detection and fixed columns
                            try:
                                uploaded_file.seek(0)
                                # Read first line to get expected column count
                                first_line = uploaded_file.readline().decode('utf-8').strip()
                                expected_cols = len(first_line.split(','))
                                uploaded_file.seek(0)
                                
                                # Read with fixed number of columns
                                df = pd.read_csv(uploaded_file, usecols=range(min(expected_cols, 32)), on_bad_lines='skip')
                                st.warning(f"⚠️ OTDR dataset uploaded with column limit ({min(expected_cols, 32)} columns)!")
                                
                            except Exception as e4:
                                # Method 5: Manual line-by-line parsing
                                try:
                                    uploaded_file.seek(0)
                                    content = uploaded_file.read().decode('utf-8')
                                    lines = content.split('\n')
                                    
                                    # Get header from first line
                                    header = lines[0].split(',')
                                    expected_cols = len(header)
                                    
                                    # Filter lines with correct number of columns
                                    valid_lines = [lines[0]]  # Keep header
                                    skipped_count = 0
                                    
                                    for i, line in enumerate(lines[1:], 1):
                                        if line.strip():  # Skip empty lines
                                            cols = line.split(',')
                                            if len(cols) == expected_cols:
                                                valid_lines.append(line)
                                            else:
                                                skipped_count += 1
                                                if skipped_count <= 5:  # Show first 5 problematic lines
                                                    st.warning(f"Skipping line {i+1}: Expected {expected_cols} columns, got {len(cols)}")
                                    
                                    if skipped_count > 5:
                                        st.warning(f"... and {skipped_count-5} more problematic lines skipped")
                                    
                                    # Create DataFrame from valid lines
                                    from io import StringIO
                                    valid_csv = '\n'.join(valid_lines)
                                    df = pd.read_csv(StringIO(valid_csv))
                                    
                                    st.success(f"✅ Manual parsing successful! Kept {len(df)} valid rows, skipped {skipped_count} problematic rows.")
                                    
                                except Exception as e5:
                                    st.error(f"❌ Failed to read CSV file after all attempts:")
                                    st.error(f"**Method 1 (Standard):** {str(e1)}")
                                    st.error(f"**Method 2 (Skip errors):** {str(e2)}")
                                    st.error(f"**Method 3 (Flexible):** {str(e3)}")
                                    st.error(f"**Method 4 (Column limit):** {str(e4)}")
                                    st.error(f"**Method 5 (Manual parsing):** {str(e5)}")
                                    
                                    st.info("💡 **CSV File Troubleshooting:**")
                                    st.info("1. **Check file format:** Ensure it's a valid CSV with consistent columns")
                                    st.info("2. **Remove extra commas:** Look for embedded commas in data fields")
                                    st.info("3. **Check encoding:** Try saving as UTF-8 CSV")
                                    st.info("4. **Validate headers:** Ensure column names are properly formatted")
                                    st.info("5. **Remove empty rows:** Delete any completely empty rows at the end")
                                    st.info("6. **Fix line 1773:** Check what's causing the extra fields in that specific line")
                                    df = None  # Set df to None to skip further processing
                
                if df is not None:
                    # Display dataset info
                    st.info(f"Dataset shape: {df.shape}")
                    
                    # Show first few rows
                    with st.expander("📋 Preview Dataset (First 5 rows)"):
                        st.dataframe(df.head())
                    
                    # Show column information
                    with st.expander("📊 Column Information"):
                        col_info = pd.DataFrame({
                            'Column': df.columns,
                            'Type': df.dtypes,
                            'Non-Null Count': df.count(),
                            'Null Count': df.isnull().sum()
                        })
                        st.dataframe(col_info)
                    
                    # Validate required columns
                    required_cols = ['SNR'] + [f'P{i}' for i in range(1, 31)]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                        st.info("📋 **Required columns for OTDR analysis:**")
                        st.info("• **SNR**: Signal-to-Noise Ratio")
                        st.info("• **P1-P30**: OTDR trace points (30 measurement points)")
                        
                        # Show available columns that might be similar
                        available_cols = list(df.columns)
                        st.info(f"📋 **Available columns ({len(available_cols)}):**")
                        col_display = ", ".join(available_cols[:10])
                        if len(available_cols) > 10:
                            col_display += f", ... and {len(available_cols)-10} more"
                        st.info(col_display)
                        
                        # Try to suggest column mappings
                        suggestions = []
                        for req_col in missing_cols[:5]:  # Show first 5 missing
                            similar_cols = [col for col in available_cols if req_col.lower() in col.lower() or col.lower() in req_col.lower()]
                            if similar_cols:
                                suggestions.append(f"• **{req_col}** might be: {', '.join(similar_cols[:3])}")
                        
                        if suggestions:
                            st.info("💡 **Possible column mappings:**")
                            for suggestion in suggestions:
                                st.info(suggestion)
                    else:
                        st.success("✅ All required columns found!")
                        
                        # Check for data quality issues
                        quality_issues = []
                        
                        # Check for missing values in required columns
                        for col in required_cols:
                            null_count = df[col].isnull().sum()
                            if null_count > 0:
                                quality_issues.append(f"Column '{col}' has {null_count} missing values")
                        
                        # Check for non-numeric values in numeric columns
                        for col in required_cols:
                            try:
                                pd.to_numeric(df[col], errors='raise')
                            except:
                                non_numeric = df[col].apply(lambda x: not str(x).replace('.','',1).replace('-','',1).isdigit()).sum()
                                if non_numeric > 0:
                                    quality_issues.append(f"Column '{col}' has {non_numeric} non-numeric values")
                        
                        if quality_issues:
                            st.warning("⚠️ **Data Quality Issues Detected:**")
                            for issue in quality_issues[:5]:  # Show first 5 issues
                                st.warning(f"• {issue}")
                            
                            if len(quality_issues) > 5:
                                st.warning(f"• ... and {len(quality_issues)-5} more issues")
                            
                            # Option to clean data
                            if st.button("🧹 Clean Dataset (Remove problematic rows)"):
                                original_rows = len(df)
                                
                                # Remove rows with missing values in required columns
                                df = df.dropna(subset=required_cols)
                                
                                # Convert columns to numeric, replacing errors with NaN
                                for col in required_cols:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                
                                # Remove rows with any NaN values after conversion
                                df = df.dropna(subset=required_cols)
                                
                                cleaned_rows = len(df)
                                removed_rows = original_rows - cleaned_rows
                                
                                if removed_rows > 0:
                                    st.success(f"✅ Dataset cleaned! Removed {removed_rows} problematic rows.")
                                    st.info(f"New dataset shape: {df.shape}")
                                else:
                                    st.info("ℹ️ No rows needed to be removed.")
                        
                        # Sample selection
                        if len(df) > 0:
                            sample_idx = st.selectbox("Select sample for analysis:", range(len(df)))
                            selected_sample = df.iloc[sample_idx]
                            
                            st.session_state.input_data = selected_sample
                            st.session_state.otdr_trace = extract_otdr_features(selected_sample)
                            
                            # Selected sample info display removed as requested
                        else:
                            st.error("❌ No valid data rows remaining after quality check!")
                
            except Exception as e:
                st.error(f"Unexpected error reading file: {str(e)}")
                st.info("💡 Please check that your file is a valid CSV format.")
    
    elif input_method == "Single OTDR Sample":
        st.subheader("Enter Single OTDR Sample")
        
        with st.form("single_sample_form"):
            snr = st.number_input("SNR", value=10.0, step=0.1, min_value=0.0, max_value=50.0)
            
            st.write("**OTDR Trace Points (P1-P30):**")
            trace_points = []
            
            # Create columns for trace points input
            cols = st.columns(3)
            for i in range(30):
                col_idx = i % 3
                with cols[col_idx]:
                    point_val = st.number_input(f"P{i+1}", value=0.5, step=0.01, key=f"p{i+1}", 
                                              min_value=0.0, max_value=1.0)
                    trace_points.append(point_val)
            
            submitted = st.form_submit_button("Submit OTDR Sample")
            
            if submitted:
                # Create sample data
                sample_data = {'SNR': snr}
                for i, val in enumerate(trace_points):
                    sample_data[f'P{i+1}'] = val
                
                sample_series = pd.Series(sample_data)
                st.session_state.input_data = sample_series
                st.session_state.otdr_trace = np.array(trace_points)
                st.success("✅ OTDR sample submitted!")
    
    else:  # Manual OTDR Input
        if st.button("Generate Sample OTDR Data"):
            # Generate realistic OTDR trace
            np.random.seed(42)
            
            # Simulate OTDR trace with possible fault
            trace_length = 30
            base_trace = np.linspace(1.0, 0.1, trace_length)  # Decreasing power
            noise = np.random.normal(0, 0.05, trace_length)
            
            # Add fault signature randomly
            fault_pos = np.random.randint(5, 25)
            fault_type = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])
            
            if fault_type > 0:  # Add fault
                if fault_type == 5:  # Fiber cut - sharp drop
                    base_trace[fault_pos:] *= 0.1
                elif fault_type in [2, 4]:  # Bad splice, dirty connector - loss
                    base_trace[fault_pos:] *= 0.7
                elif fault_type == 7:  # Reflector - spike
                    base_trace[fault_pos] += 0.3
            
            otdr_trace = base_trace + noise
            otdr_trace = np.clip(otdr_trace, 0, 1)  # Normalize
            
            # Create sample
            sample_data = {'SNR': np.random.uniform(8, 15)}
            for i, val in enumerate(otdr_trace):
                sample_data[f'P{i+1}'] = val
            
            sample_series = pd.Series(sample_data)
            st.session_state.input_data = sample_series
            st.session_state.otdr_trace = otdr_trace
            st.success("✅ Sample OTDR data generated!")

# Display OTDR trace visualization
if hasattr(st.session_state, 'otdr_trace') and st.session_state.otdr_trace is not None:
    st.subheader("📊 OTDR Trace Visualization")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, 31)),
        y=st.session_state.otdr_trace,
        mode='lines+markers',
        name='OTDR Trace',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="OTDR Trace (P1-P30)",
        xaxis_title="Position Index",
        yaxis_title="Normalized Power",
        height=300,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("🔄 ML Processing Pipeline")
    
    # Check if binary model is loaded
    if not models_status['binary']:
        st.markdown('<div class="warning-box">⚠️ <strong>Binary Classification Model Required</strong><br>Please upload a binary classification model to start fault detection.</div>', unsafe_allow_html=True)
    
    if hasattr(st.session_state, 'input_data') and st.session_state.input_data is not None:
        # Step 1: Binary Classification
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.subheader("Step 1: Binary Fault Detection")
        st.write("*Determines if there is any fault in the fiber (Normal vs Fault)*")
        
        if models_status['binary']:
            # Check if scaler is available
            if st.session_state.scaler is None:
                st.markdown('<div class="warning-box">⚠️ <strong>Feature Scaler Not Loaded</strong><br>For best results, upload the StandardScaler used during training. Predictions will use raw features without normalization.</div>', unsafe_allow_html=True)
            
            if st.button("🔍 Run Binary Classification", key="binary_btn"):
                with st.spinner("Running binary classification..."):
                    # Prepare input for binary model
                    if debug_mode:
                        st.write("**Preprocessing OTDR data for binary classification...**")
                    input_features = preprocess_for_model(st.session_state.input_data, debug=debug_mode, model=st.session_state.binary_model, scaler=st.session_state.scaler, prediction_type='binary')
                    
                    if input_features is not None:
                        if debug_mode:
                            st.write("**Making prediction with binary model...**")
                            
                            # Show model info
                            model = st.session_state.loaded_models['binary']
                            if hasattr(model, 'input_shape'):
                                st.write(f"**Model expects input shape:** {model.input_shape}")
                            elif hasattr(model, 'n_features_in_'):
                                st.write(f"**Model expects {model.n_features_in_} features**")
                        
                        # Get prediction
                        result = predict_with_model(st.session_state.loaded_models['binary'], input_features, 'binary', debug=debug_mode)
                        if result is not None:
                            st.session_state.binary_prediction = result
                            st.success("✅ Binary classification completed successfully!")
                        else:
                            st.error("❌ Failed to get prediction from binary model")
                    else:
                        st.error("❌ Failed to preprocess input data for binary model")
        else:
            st.warning("⚠️ Binary classification model not loaded. Please upload a model first.")
            st.info("📋 **Expected model input:** 31 features (1 SNR + 30 OTDR trace points P1-P30)")
            st.info("📋 **Model types supported:** .pkl (scikit-learn), .joblib (scikit-learn), .h5 (Keras/TensorFlow)")
            st.info("📋 **Binary output:** 0 = No Fault, 1 = Fault Detected")
        
        if hasattr(st.session_state, 'binary_prediction') and st.session_state.binary_prediction is not None:
            result = st.session_state.binary_prediction
            
            if result['prediction']:
                st.markdown(f'<div class="error-box">🚨 <strong>FAULT DETECTED</strong><br>Confidence: {result["confidence"]:.3f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box">✅ <strong>NO FAULT DETECTED</strong><br>Confidence: {result["confidence"]:.3f}</div>', unsafe_allow_html=True)

            # QoL: Show binary prediction as a metric card
            col_metric, _ = st.columns([1, 3])
            with col_metric:
                metric_label = "Binary Fault"
                metric_value = "Fault Detected" if result['prediction'] else "No Fault"
                metric_delta = f"Confidence: {result['confidence']:.3f}"
                st.metric(metric_label, metric_value, metric_delta)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 2: Detailed Analysis (only if fault detected and binary prediction exists)
        if (hasattr(st.session_state, 'binary_prediction') and 
            st.session_state.binary_prediction is not None and 
            st.session_state.binary_prediction['prediction']):
            
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.subheader("Step 2: Detailed Fault Analysis")
            st.write("*Advanced analysis for fault classification, localization, and characterization*")
            
            # Check which detailed models are available
            available_models = []
            if models_status['class']:
                available_models.append("Fault Classification")
            if models_status['position']:
                available_models.append("Position Localization")
            if models_status['reflectance']:
                available_models.append("Reflectance Analysis")
            if models_status['loss']:
                available_models.append("Loss Analysis")
            
            if available_models:
                st.info(f"Available analysis: {', '.join(available_models)}")
                
                if st.button("🔬 Run Detailed Analysis", key="detailed_btn"):
                    with st.spinner("Running detailed analysis..."):
                        input_features = preprocess_for_model(st.session_state.input_data, model=st.session_state.loaded_models.get('class'), scaler=st.session_state.scaler, prediction_type='class')
                        
                        if input_features is not None:
                            predictions = {}
                            
                            # Fault Class Detection
                            if models_status['class']:
                                class_result = predict_with_model(st.session_state.loaded_models['class'], input_features, 'class')
                                if class_result is not None:
                                    predictions['class'] = {
                                        'value': int(class_result['prediction']),
                                        'name': FAULT_CLASSES.get(int(class_result['prediction']), 'Unknown')
                                    }
                            
                            # Position Localization  
                            if models_status['position']:
                                input_features = preprocess_for_model(st.session_state.input_data, model=st.session_state.loaded_models.get('position'), scaler=st.session_state.scaler, prediction_type='position')
                                pos_result = predict_with_model(st.session_state.loaded_models['position'], input_features, 'position')
                                if pos_result is not None:
                                    predictions['position'] = {
                                        'value': float(pos_result['prediction']),
                                        'distance_km': float(pos_result['prediction']) * 100  # Convert to km (assuming 100km max)
                                    }
                            
                            # Reflectance Analysis
                            if models_status['reflectance']:
                                input_features = preprocess_for_model(st.session_state.input_data, model=st.session_state.loaded_models.get('reflectance'), scaler=st.session_state.scaler, prediction_type='reflectance')
                                refl_result = predict_with_model(st.session_state.loaded_models['reflectance'], input_features, 'reflectance')
                                if refl_result is not None:
                                    predictions['reflectance'] = {
                                        'value': float(refl_result['prediction'])
                                    }
                            
                            # Loss Analysis
                            if models_status['loss']:
                                input_features = preprocess_for_model(st.session_state.input_data, model=st.session_state.loaded_models.get('loss'), scaler=st.session_state.scaler, prediction_type='loss')
                                loss_result = predict_with_model(st.session_state.loaded_models['loss'], input_features, 'loss')
                                if loss_result is not None:
                                    predictions['loss'] = {
                                        'value': float(loss_result['prediction'])
                                    }
                            
                            if predictions:
                                st.session_state.detailed_predictions = predictions
                            else:
                                st.error("No predictions could be generated from available models")
            else:
                st.warning("⚠️ No detailed analysis models loaded. Upload models for fault classification, position localization, reflectance, or loss analysis.")
            
            # Display detailed results
            if hasattr(st.session_state, 'detailed_predictions') and st.session_state.detailed_predictions is not None:
                preds = st.session_state.detailed_predictions
                
                # Create metrics display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'class' in preds:
                        st.metric(
                            "Fault Class",
                            f"{preds['class']['value']}: {preds['class']['name']}",
                            "Predicted"
                        )
                    else:
                        st.metric("Fault Class", "N/A", "Model not loaded")
                
                with col2:
                    if 'position' in preds:
                        st.metric(
                            "Position",
                            f"{preds['position']['value']:.3f}"
                        )
                    else:
                        st.metric("Position", "N/A", "Model not loaded")
                
                with col3:
                    if 'reflectance' in preds:
                        st.metric(
                            "Reflectance",
                            f"{preds['reflectance']['value']:.3f}",
                            "Normalized"
                        )
                    else:
                        st.metric("Reflectance", "N/A", "Model not loaded")
                
                with col4:
                    if 'loss' in preds:
                        st.metric(
                            "Loss",
                            f"{preds['loss']['value']:.3f}",
                            "Normalized"
                        )
                    else:
                        st.metric("Loss", "N/A", "Model not loaded")
                
                # New: OTDR Trace with Predicted Fault Position
                if hasattr(st.session_state, 'otdr_trace') and st.session_state.otdr_trace is not None and 'position' in preds:
                    positions = list(range(1, 31))
                    values = list(st.session_state.otdr_trace)

                    # Map prediction to position per spec: position_index = int(prediction * 100)
                    raw_prediction = float(preds['position']['value'])
                    position_index = int(raw_prediction * 100)
                    # Clamp to valid range p1..p30
                    fault_pos_x = max(1, min(30, position_index))
                    fault_y = values[fault_pos_x - 1]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=positions,
                        y=values,
                        mode='lines+markers',
                        name='OTDR Trace',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=5)
                    ))

                    # Highlight predicted fault position
                    fig.add_trace(go.Scatter(
                        x=[fault_pos_x],
                        y=[fault_y],
                        mode='markers+text',
                        name='Predicted Fault',
                        marker=dict(color='red', size=14, symbol='diamond'),
                        text=["Fault Here"],
                        textposition="top center"
                    ))

                    fig.update_layout(
                        title="OTDR Trace with Predicted Fault Position",
                        xaxis_title="Position",
                        yaxis_title="Value",
                        height=400,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # QoL: Quick download of predictions (CSV)
                try:
                    export_data = {
                        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        'SNR': [st.session_state.input_data['SNR']],
                        'Binary_Prediction': [st.session_state.binary_prediction['prediction']],
                        'Binary_Confidence': [st.session_state.binary_prediction['confidence']],
                    }
                    if 'class' in preds:
                        export_data['Fault_Class'] = [preds['class']['value']]
                        export_data['Fault_Name'] = [preds['class']['name']]
                    if 'position' in preds:
                        export_data['Position'] = [preds['position']['value']]
                        export_data['Position_km'] = [preds['position'].get('distance_km', None)]
                    if 'reflectance' in preds:
                        export_data['Reflectance'] = [preds['reflectance']['value']]
                    if 'loss' in preds:
                        export_data['Loss'] = [preds['loss']['value']]

                    # Add OTDR trace points
                    if hasattr(st.session_state, 'otdr_trace') and st.session_state.otdr_trace is not None:
                        for i, val in enumerate(st.session_state.otdr_trace):
                            export_data[f'P{i+1}'] = [val]

                    quick_export_df = pd.DataFrame(export_data)
                    quick_csv = quick_export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Quick Download Predictions (CSV)",
                        data=quick_csv,
                        file_name=f"otdr_quick_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception:
                    pass
            
            st.markdown('</div>', unsafe_allow_html=True)

# Results Analysis Section
if hasattr(st.session_state, 'detailed_predictions') and st.session_state.detailed_predictions is not None:
    st.header("📊 Analysis Results & Performance")
    
    tab2, tab3 = st.tabs(["📋 Detailed Report", "📥 Export Results"])
    
    with tab2:
        st.subheader("📋 Comprehensive Analysis Report")
        
        # Generate detailed report
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        preds = st.session_state.detailed_predictions
        
        report = f"""
        # OTDR Fiber Fault Analysis Report
        
        **Analysis Timestamp:** {report_time}
        **Sample SNR:** {st.session_state.input_data['SNR']:.3f}
        
        ## Binary Classification Results
        - **Fault Detection:** {'FAULT DETECTED' if st.session_state.binary_prediction['prediction'] else 'NO FAULT'}
        - **Confidence:** {st.session_state.binary_prediction['confidence']:.3f}
        
        ## Detailed Fault Analysis
        """
        
        if 'class' in preds:
            report += f"""
        ### Fault Classification
        - **Predicted Class:** {preds['class']['value']} ({preds['class']['name']})
        - **Class Description:** {FAULT_CLASSES.get(preds['class']['value'], 'Unknown fault type')}
        """
        
        if 'position' in preds:
            report += f"""
        ### Fault Localization
        - **Position Index:** {preds['position']['value']:.3f}
        - **Estimated Distance:** {preds['position']['distance_km']:.1f} km (assuming 100km total span)
        - **Trace Point:** P{int(preds['position']['value'] * 30) + 1}
        """
        
        report += f"""
        ### Fault Characteristics
        - **Reflectance:** {f"{preds['reflectance']['value']:.3f} (normalized)" if 'reflectance' in preds else 'N/A (model not loaded)'}
        - **Loss:** {f"{preds['loss']['value']:.3f} (normalized)" if 'loss' in preds else 'N/A (model not loaded)'}
        
        ## OTDR Trace Analysis
        - **Trace Length:** 30 points (P1-P30)
        - **Signal Quality:** Based on SNR of {st.session_state.input_data['SNR']:.3f}
        """
        
        if 'class' in preds:
            report += f"""
        ## Recommendations
        Based on the detected fault type ({preds['class']['name']}):
        """
            
            # Add specific recommendations based on fault type
            fault_type = preds['class']['value']
            if fault_type == 1:  # Fiber Tapping
                report += "\n- Immediate security inspection required\n- Check for unauthorized access points\n- Consider fiber encryption"
            elif fault_type == 2:  # Bad Splice
                report += "\n- Schedule splice re-work\n- Verify splice loss specifications\n- Consider fusion splice instead of mechanical"
            elif fault_type == 3:  # Bending Event
                report += "\n- Check cable routing and support\n- Verify minimum bend radius compliance\n- Inspect for physical stress points"
            elif fault_type == 4:  # Dirty Connector
                report += "\n- Clean connectors with appropriate cleaning tools\n- Inspect end faces under microscope\n- Re-test after cleaning"
            elif fault_type == 5:  # Fiber Cut
                report += "\n- CRITICAL: Complete fiber break detected\n- Emergency repair required\n- Activate backup routes if available"
            elif fault_type == 6:  # PC Connector
                report += "\n- Verify connector type and specifications\n- Check insertion loss\n- Consider APC connectors if applicable"
            elif fault_type == 7:  # Reflector
                report += "\n- Identify source of reflection\n- Check for unterminated fibers\n- Verify proper connector end face preparation"
        
        report += f"""
        
        ## Technical Notes
        - Analysis performed using uploaded ML models
        - Position accuracy depends on OTDR resolution and model training
        - Normalized values require denormalization for absolute measurements
        - Consider environmental factors during maintenance planning
        - Models loaded: {', '.join([k for k, v in models_status.items() if v])}
        
        ---
        *Report generated by AI-Driven OTDR Fault Detection System*
        """
        
        st.markdown(report)
    
    with tab3:
        st.subheader("📥 Export Analysis Results")
        
        if st.button("Generate CSV Export"):
            # Create export data
            export_data = {
                'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'SNR': [st.session_state.input_data['SNR']],
                'Binary_Prediction': [st.session_state.binary_prediction['prediction']],
                'Binary_Confidence': [st.session_state.binary_prediction['confidence']],
            }
            
            # Add detailed predictions
            if 'class' in st.session_state.detailed_predictions:
                export_data['Fault_Class'] = [st.session_state.detailed_predictions['class']['value']]
                export_data['Fault_Name'] = [st.session_state.detailed_predictions['class']['name']]
            else:
                export_data['Fault_Class'] = ['N/A']
                export_data['Fault_Name'] = ['N/A']
            
            if 'position' in st.session_state.detailed_predictions:
                export_data['Position'] = [st.session_state.detailed_predictions['position']['value']]
            else:
                export_data['Position'] = ['N/A']
            
            if 'reflectance' in st.session_state.detailed_predictions:
                export_data['Reflectance'] = [st.session_state.detailed_predictions['reflectance']['value']]
            else:
                export_data['Reflectance'] = ['N/A']
            
            if 'loss' in st.session_state.detailed_predictions:
                export_data['Loss'] = [st.session_state.detailed_predictions['loss']['value']]
            else:
                export_data['Loss'] = ['N/A']
            
            # Add OTDR trace points
            for i, val in enumerate(st.session_state.otdr_trace):
                export_data[f'P{i+1}'] = [val]
            
            export_df = pd.DataFrame(export_data)
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"otdr_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("🔬 **OTDR-Based Fiber Fault Detection System** | Built with Streamlit for ML Model Integration")
st.markdown("*Dataset Format: SNR + 30 OTDR Trace Points (P1-P30) + Fault Classification & Localization*")
st.markdown("**Note:** This system requires trained ML models to be uploaded for predictions. No simulated results are provided.")
