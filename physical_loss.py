import numpy as np

def cart2sph_just_elev(x,y,z):
  hypotxy = np.hypot(x,y)
  elev    = np.arctan2(z,hypotxy)
  return elev

class ZeroIncidencePhysicalLoss:

  def __init__(self, sz=10, limit_mask=-9):
    
    self.limit_mask = limit_mask
    
    c     = 3e8       # Velocidad de la luz
    freq  = 25e9      # Frecuencia de trabajo
    lambd = c/freq    # Longitud de onda
    k     = (2*np.pi)/lambd

    # Feed
    theta_feed = 0*np.pi/180    # Angulo en theta del feed
    phi_feed   = 0*np.pi/180    # Angulo rn phi del feed
    d_foco     = 15000e-3       # Campo lejano 
    x_feed     = np.sin(theta_feed)*np.cos(phi_feed)*d_foco # Posición-X del feed (metros)
    y_feed     = np.sin(theta_feed)*np.sin(phi_feed)*d_foco # Posición-Y del feed (metros)
    z_feed     = np.cos(theta_feed)*d_foco                  # Posición-Z del feed (metros)
    qfeed      = 27                         # Factor del feed para la amplitud del campo
    qelem      = 1                          # Factor del elemento para la amplitud del campo
    Gamma      = 1                          # Coeficiente de reflexión del RA
    self.mX    = sz     # No. de elementos en la dirección-X
    self.mY    = sz     # No. de elementos en la dirección-Y
    perX       = 9e-3 # Distancia entre elementos en la dirección-X (PERIODICIDAD)
    perY       = 9e-3 # Distancia entre elementos en la dirección-Y (PERIODICIDAD)

    # Matriz de posiciones
                               # Iniciación matriz (1ra dimensión: posición en X.
                               #                    2da dimensión: posición en Y.
                               #                    3ra dimensión: posición en Z.)
    R        = np.zeros((self.mX,self.mY,3))
    PosEleX  = np.linspace(-(self.mX-1)/2,  (self.mX-1)/2, num=self.mX, endpoint=True) * perX # Posición de los elementos (en el eje X)
    PosEleY  = np.linspace( (self.mY-1)/2, -(self.mY-1)/2, num=self.mY, endpoint=True) * perY # Posición de los elementos (en el eje Y)
    R[:,:,0] = PosEleX.reshape(( 1,-1)) # Matriz de posiciones de X
    R[:,:,1] = PosEleY.reshape((-1, 1)) # Matriz de posiciones de Y
    R_i      = np.sqrt((R[:,:,0]-x_feed)**2 + (R[:,:,1]-y_feed)**2 + (R[:,:,2]-z_feed)**2)

    self.step_phi     = 181
    self.step_theta   = 361
    v_phi             = np.linspace(0,np.pi,self.step_phi)
    v_theta           = np.linspace(-np.pi/2,np.pi/2,self.step_theta)
    r_mn              = R
    matrix_phi        = np.empty((self.step_phi, self.step_theta), dtype=v_phi.dtype)
    matrix_phi[:,:]   = v_phi.reshape((-1,1))
    matrix_theta      = np.empty((self.step_phi, self.step_theta), dtype=v_phi.dtype)
    matrix_theta[:,:] = v_theta.reshape((1,-1))
    #matrix_phi = repmat(v_phi',1,step_theta);
    #matrix_theta = repmat(v_theta,step_phi,1);
    #x1=np.arange(10); r=np.empty((10,10),dtype=x1.dtype); r[:,:]=x1.reshape((-1,1)); r
    #x1=np.arange(10).reshape((-1,1))
    
    sin_matrix_theta   = np.sin(matrix_theta)
    cos_matrix_theta   = np.cos(matrix_theta)
    cos_matrix_phi     = np.cos(matrix_phi)
    sin_matrix_phi     = np.sin(matrix_phi)
    sin_theta_cos_phi  = sin_matrix_theta * cos_matrix_phi
    sin_theta_sin_phi  = sin_matrix_theta * sin_matrix_phi
    r_e_f              = [x_feed-r_mn[:,:,0], y_feed-r_mn[:,:,1], z_feed-r_mn[:,:,2]]
    theta_e_mn         = cart2sph_just_elev(*r_e_f)
    tau_mn             = np.power(np.cos(np.pi/2 - theta_e_mn), qelem) #Modeled by a cosine model whose pointing angle is pi/2 (z-direction)
    # Element excitation function: I_mn
    theta_f_center     = cart2sph_just_elev(x_feed, y_feed, z_feed)
    #r_f_e             = [-x for x in r_e_f]
    theta_f_mn         = cart2sph_just_elev(*r_e_f) #*r_f_e
    #Radiation pattern of the feed modeled by a cosine model
    #whose pointing angle (theta_f_center) is at the center of the RA (0,0,0)
    angle              = -k*R_i
    I_mn_constant      = np.power(np.cos(theta_f_center - theta_f_mn), qfeed) * (np.cos(angle)+np.sin(angle)*1j) * tau_mn
    self.aux_mn        = np.zeros((self.mX, self.mY, 1, self.step_phi, self.step_theta), dtype=np.complex64)
    for ii in range(self.mX):
      for jj in range(self.mY):
        dot_rmn_uo               = r_mn[ii,jj,0]*sin_theta_cos_phi + r_mn[ii,jj,1]*sin_theta_sin_phi + r_mn[ii,jj,2]*cos_matrix_theta
        angle                    = k*dot_rmn_uo
        A_mn                     = np.power(cos_matrix_theta, qelem) * (np.cos(angle)+np.sin(angle)*1j)
        self.aux_mn[ii,jj,0,:,:] = A_mn * I_mn_constant[ii,jj]

    self.     np_sin     =    np.sin
    self.     np_cos     =    np.cos
    self.     np_log10   =    np.log10
    self.     np_maximum =    np.maximum
    self.     np_abs     =    np.abs
    self.     np_power   =    np.power
    self.     np_where   =    np.where
    #self.     np_take    =    np.take

  def compute_fitness(self, prediction, real_grid):
    if True:
      sin     = self.   np_sin
      cos     = self.   np_cos
      log10   = self.   np_log10
      maximum = self.   np_maximum
      m_abs   = self.   np_abs
      power   = self.   np_power
      where   = self.   np_where
      #take    = self.   np_take
      aux_mn  = self.   aux_mn
      zero    = 0
      lmask   = self.limit_mask
      #cmplx   = self.complexes
    lp = len(prediction.shape)
    lr = len(real_grid .shape)
    if   lp==3 and lr==3:
      pass #everything ok
    elif lp==3 and lr==4:
      real_grid = real_grid[:,0,:,:]
    elif lp==2 and lr==2:
      prediction = np.expand_dims(prediction, 0)
      real_grid  = np.expand_dims(real_grid,  0)
    elif lp==2 and lr==3:
      prediction = np.expand_dims(prediction, 0)
      real_grid  = np.expand_dims(real_grid[0,:,:],  0)
    else:
      raise Exception(f'Unrecognized shapes for prediction and real_grid: {lp} and {lr}')

    if True:
      #nbatch    = prediction.shape[0]
      E_pattern_real   = 0 # np.zeros((nbatch, self.step_phi,self.step_theta), dtype=np.complex64)
      E_pattern_salida = 0 # np.zeros((nbatch, self.step_phi,self.step_theta), dtype=np.complex64)

      angle_real      = real_grid  * np.pi #       grid has 0-1 values, translated to 0-pi angles
      angle_salida    = prediction * np.pi # prediction has 0-1 values, translated to 0-pi angles
      complejo_real   = cos(angle_real  )+sin(angle_real  )*1j
      complejo_salida = cos(angle_salida)+sin(angle_salida)*1j
      ######### PARA TERMINAR DE IMPLEMENTAR ESTA OPTIMIZACION: HAY QUE SACAR symbolic_grid DEL CONJUNTO DE VALIDACION (QUIZA USANDO additional), Y LA CORRESPONDIENTE DE LA PREDICCION (SE CALCULA/PUEDE CALCULAR DE FORMA BARATA EN VALIDACION)
      #if compute_sincos:
      #  angle_real      = real_grid  * np.pi #       grid has 0-1 values, translated to 0-pi angles
      #  angle_salida    = prediction * np.pi # prediction has 0-1 values, translated to 0-pi angles
      #  complejo_real   = cos(angle_real  )+sin(angle_real  )*1j
      #  complejo_salida = cos(angle_salida)+sin(angle_salida)*1j
      #else:
      #  complejo_real   = take(real_grid,  cmplx)
      #  complejo_salida = take(prediction, cmplx)
      for ii in range(self.mX):
        for jj in range(self.mY):
          aux_real         = aux_mn[ii,jj] * complejo_real  [:,ii,jj].reshape((-1,1,1))
          aux_salida       = aux_mn[ii,jj] * complejo_salida[:,ii,jj].reshape((-1,1,1))
          E_pattern_real   = E_pattern_real     + aux_real
          E_pattern_salida = E_pattern_salida   + aux_salida

      #Dibujo en esféricas
      FA_power_dB_real   = 20*log10(m_abs(E_pattern_real))
      FA_power_dB_salida = 20*log10(m_abs(E_pattern_salida))
      # NOTA DE TRADUCCION: Esto creo que no es estrictamente necesario, porque debería quedar subsumido en las operaciones con EdB de abajo
      FA_power_dB_real   = maximum(FA_power_dB_real,   zero) #truncamos por debajo del 0 para que el plot en esféricas no salga mal...
      FA_power_dB_salida = maximum(FA_power_dB_salida, zero) #truncamos por debajo del 0 para que el plot en esféricas no salga mal...

      # Esta función realiza el calculo del error de un diagrama de radiación en comparación con el real (para compara un 
      # y otro). 
      # Dado un limite inferior (e.g. -9dB), contruye una máscara la cuál limita a -9dB para cuando el diagrama esta por
      # debajo de ese nivel, y para el caso de ser superior realiza la forma de la máscara con la misma forma que el diagrama
      # real. Después realiza la comparación con el diagrama modificado y calcula el error entre este último y la mácara.
      #
      # Variables de entrada:
      #     FA: campo electrico tanto del generado con el tablero original y under test
      #     limit_mask: El límite del cual se desea partir para la comparación
      # Variables de salida: 
      #     error: Error cuadrático calculado como el cuadrado de la diferencia entre el valor de la máscara y el campo
      #            calculado con el tablero 'modificado' divido entre n.
      max_EdB = FA_power_dB_real.reshape(FA_power_dB_real.shape[0], -1).max(axis=-1)[0]
      max_EdB = max_EdB.reshape(-1, 1, 1)
      EdB     = FA_power_dB_real - max_EdB
      mask    = maximum(EdB, lmask)
      
      EdB_modificado = FA_power_dB_salida - max_EdB
      field          = maximum(EdB_modificado, lmask)

      bool_mask      = field!=mask

      errors         = where(bool_mask, power(m_abs(mask-field), 2), zero)
      
      radiation_error= errors.sum()/(181*361)



    return bool_mask, errors, radiation_error
